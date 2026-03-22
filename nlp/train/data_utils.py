"""Dataset classes and dataloader factory for multi-task mental-health training.

Two dataset types:
  MentalHealthDataset   — reads domain JSONL (all four task labels present)
  PublicEmotionDataset  — reads public-dataset JSONL (emotion label only)

Usage:
    from nlp.train.data_utils import MentalHealthDataset, build_dataloaders
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from nlp.train.model import EMOTION_LABELS, INTENT_LABELS

_EMOTION_INDEX: dict[str, int] = {label: i for i, label in enumerate(EMOTION_LABELS)}
_INTENT_INDEX: dict[str, int] = {label: i for i, label in enumerate(INTENT_LABELS)}

MAX_LENGTH = 128


class MentalHealthDataset(Dataset):
    """Domain JSONL dataset — all four tasks labelled."""

    def __init__(self, path: str | Path, tokenizer_name: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._records: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self._records[idx]
        enc = self._tokenizer(
            rec["text"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "emotion_label": torch.tensor(_EMOTION_INDEX[rec["emotion_label"]], dtype=torch.long),
            "intent_label": torch.tensor(_INTENT_INDEX[rec["intent_label"]], dtype=torch.long),
            "intensity_score": torch.tensor(float(rec["intensity_score"]), dtype=torch.float32),
            "risk_flag": torch.tensor(1.0 if rec["risk_flag"] else 0.0, dtype=torch.float32),
        }


class PublicEmotionDataset(Dataset):
    """Public-dataset JSONL — emotion label only (no intent / intensity / risk)."""

    def __init__(self, path: str | Path, tokenizer_name: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._records: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self._records[idx]
        enc = self._tokenizer(
            rec["text"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "emotion_label": torch.tensor(_EMOTION_INDEX[rec["emotion_label"]], dtype=torch.long),
        }


def build_dataloaders(
    train_path: str | Path,
    dev_path: str | Path,
    batch_size: int,
    tokenizer_name: str,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, dev_loader) for the domain dataset."""
    train_ds = MentalHealthDataset(train_path, tokenizer_name)
    dev_ds = MentalHealthDataset(dev_path, tokenizer_name)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, dev_loader
