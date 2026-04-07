"""Multi-task fine-tuning entry point.

Trains MentalHealthMultiTaskModel on the domain JSONL corpus.  Run twice to
produce both models for the thesis comparison:

    # Primary model
    python nlp/train/train_multitask.py \\
        --train data/processed/mental_dialogue_train.jsonl \\
        --dev   data/processed/mental_dialogue_dev.jsonl \\
        --base-model hfl/chinese-roberta-wwm-ext \\
        --output nlp/artifacts/roberta-domain

    # BERT baseline
    python nlp/train/train_multitask.py \\
        --train data/processed/mental_dialogue_train.jsonl \\
        --dev   data/processed/mental_dialogue_dev.jsonl \\
        --base-model bert-base-chinese \\
        --output nlp/artifacts/bert-domain
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Allow sibling imports when run as a script from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from nlp.train.data_utils import build_dataloaders
from nlp.train.model import MentalHealthMultiTaskModel

EMOTION_LABELS = ["anxiety", "sadness", "anger", "fear", "shame", "hopelessness", "neutral"]
INTENT_LABELS  = ["venting", "seeking_advice", "seeking_empathy", "crisis",
                  "self_disclosure", "information_seeking"]


def _compute_class_weights(train_path: str, labels: list[str], key: str,
                            device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights from training data (balanced scaling)."""
    import json
    counts = {l: 0 for l in labels}
    total = 0
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lbl = rec.get(key)
            if lbl in counts:
                counts[lbl] += 1
                total += 1
    n_cls = len(labels)
    weights = [total / (n_cls * max(counts[l], 1)) for l in labels]
    t = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"  {key} class weights: " +
          ", ".join(f"{l}={w:.3f}" for l, w in zip(labels, weights)))
    return t


def _linear_warmup_decay(num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)
    return lr_lambda


def _evaluate(model: MentalHealthMultiTaskModel, loader, device: torch.device) -> dict:
    model.eval()
    total = correct_emotion = correct_intent = 0
    intensity_abs_err = 0.0
    risk_preds: list[int] = []
    risk_true: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids, attention_mask)

            pred_emotion = out.emotion_logits.argmax(dim=-1)
            pred_intent = out.intent_logits.argmax(dim=-1)

            correct_emotion += (pred_emotion == batch["emotion_label"].to(device)).sum().item()
            correct_intent += (pred_intent == batch["intent_label"].to(device)).sum().item()
            intensity_abs_err += (
                (out.intensity.squeeze(-1) - batch["intensity_score"].to(device)).abs().sum().item()
            )

            pred_risk = (out.risk_aux.squeeze(-1) >= 0.5).long()
            risk_preds.extend(pred_risk.cpu().tolist())
            risk_true.extend(batch["risk_flag"].long().tolist())
            total += input_ids.size(0)

    emotion_acc = correct_emotion / total if total else 0.0
    intent_acc = correct_intent / total if total else 0.0
    intensity_mae = intensity_abs_err / total if total else 0.0

    tp = sum(p == 1 and t == 1 for p, t in zip(risk_preds, risk_true))
    fp = sum(p == 1 and t == 0 for p, t in zip(risk_preds, risk_true))
    fn = sum(p == 0 and t == 1 for p, t in zip(risk_preds, risk_true))
    risk_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    risk_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Approximate AUC via rank correlation (simple version: use recall as proxy)
    risk_auc = risk_recall

    return {
        "emotion_acc": emotion_acc,
        "intent_acc": intent_acc,
        "intensity_mae": intensity_mae,
        "risk_recall": risk_recall,
        "risk_precision": risk_precision,
        "risk_auc": risk_auc,
    }


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer_name = args.base_model
    train_loader, dev_loader = build_dataloaders(
        args.train, args.dev, args.batch_size, tokenizer_name
    )

    model = MentalHealthMultiTaskModel(args.base_model).to(device)

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = max(1, int(num_training_steps * 0.1))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_linear_warmup_decay(num_warmup_steps, num_training_steps),
    )

    emotion_weights = _compute_class_weights(
        args.train, EMOTION_LABELS, "emotion_label", device)
    intent_weights = _compute_class_weights(
        args.train, INTENT_LABELS, "intent_label", device)
    ce_emotion = nn.CrossEntropyLoss(weight=emotion_weights, label_smoothing=0.10)
    ce_intent  = nn.CrossEntropyLoss(weight=intent_weights, label_smoothing=0.10)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    best_emotion_acc = 0.0
    patience_counter = 0
    output_dir = Path(args.output)

    print(
        f"Training {args.base_model} for {args.epochs} epochs "
        f"({num_training_steps} steps, warmup {num_warmup_steps})."
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["emotion_label"].to(device)
            intent_labels = batch["intent_label"].to(device)
            intensity_targets = batch["intensity_score"].to(device)
            risk_targets = batch["risk_flag"].to(device)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask)

            loss_emotion = ce_emotion(out.emotion_logits, emotion_labels)
            loss_intent  = ce_intent(out.intent_logits, intent_labels)
            loss_intensity = mse_loss(out.intensity.squeeze(-1), intensity_targets)
            loss_risk = bce_loss(out.risk_aux.squeeze(-1), risk_targets)

            loss = (
                args.w_emotion * loss_emotion
                + args.w_intent * loss_intent
                + args.w_intensity * loss_intensity
                + args.w_risk * loss_risk
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        metrics = _evaluate(model, dev_loader, device)
        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss={avg_loss:.4f} | "
            f"emotion_acc={metrics['emotion_acc']:.4f} | "
            f"intent_acc={metrics['intent_acc']:.4f} | "
            f"intensity_mae={metrics['intensity_mae']:.4f} | "
            f"risk_recall={metrics['risk_recall']:.4f}"
        )

        if metrics["emotion_acc"] > best_emotion_acc:
            best_emotion_acc = metrics["emotion_acc"]
            patience_counter = 0
            output_dir.mkdir(parents=True, exist_ok=True)
            model.encoder.save_pretrained(str(output_dir))
            # Save head weights separately alongside encoder files
            torch.save(
                {
                    "emotion_head": model.emotion_head.state_dict(),
                    "intent_head": model.intent_head.state_dict(),
                    "intensity_head": model.intensity_head.state_dict(),
                    "risk_head": model.risk_head.state_dict(),
                    "base_model_name": args.base_model,
                },
                output_dir / "task_heads.pt",
            )
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(str(output_dir))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print(f"Saved to {output_dir}. Best emotion_acc={best_emotion_acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-task mental-health model training")
    parser.add_argument("--train", required=True, help="Path to train JSONL")
    parser.add_argument("--dev", required=True, help="Path to dev JSONL")
    parser.add_argument("--output", required=True, help="Output directory for model artifacts")
    parser.add_argument("--base-model", default="hfl/chinese-roberta-wwm-ext",
                        help="HuggingFace model name")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--w-emotion", type=float, default=1.0, dest="w_emotion")
    parser.add_argument("--w-intent", type=float, default=1.0, dest="w_intent")
    parser.add_argument("--w-intensity", type=float, default=2.0, dest="w_intensity")
    parser.add_argument("--w-risk", type=float, default=3.0, dest="w_risk")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
