"""Model evaluation script — domain, public, and probe modes.

Domain mode  (all four tasks, human-annotated test split):
    python nlp/eval/evaluate.py \\
        --model nlp/artifacts/roberta-domain \\
        --test  data/processed/mental_dialogue_test.jsonl \\
        --mode  domain \\
        --output nlp/artifacts/roberta-domain/domain_results.json

Public mode  (emotion only, converted public dataset):
    python nlp/eval/evaluate.py \\
        --model nlp/artifacts/roberta-domain \\
        --test  data/processed/weibo_emotion_eval.jsonl \\
        --mode  public \\
        --output nlp/artifacts/roberta-domain/public_results.json

Probe mode  (30-sentence crisis probe set, pass/fail per tier):
    python nlp/eval/evaluate.py \\
        --model nlp/artifacts/roberta-domain \\
        --test  data/crisis_probe.jsonl \\
        --mode  probe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow sibling imports when run from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from nlp.train.data_utils import MentalHealthDataset, PublicEmotionDataset
from nlp.train.model import (
    EMOTION_LABELS,
    INTENT_LABELS,
    MentalHealthMultiTaskModel,
)


def _load_model(model_dir: Path) -> tuple[MentalHealthMultiTaskModel, str]:
    heads = torch.load(model_dir / "task_heads.pt", map_location="cpu")
    base_model_name: str = heads["base_model_name"]
    # Load the fine-tuned encoder from model_dir (saved by train_multitask.py),
    # not the original pre-trained weights from HuggingFace.
    model = MentalHealthMultiTaskModel(str(model_dir))
    model.emotion_head.load_state_dict(heads["emotion_head"])
    model.intent_head.load_state_dict(heads["intent_head"])
    model.intensity_head.load_state_dict(heads["intensity_head"])
    model.risk_head.load_state_dict(heads["risk_head"])
    model.eval()
    return model, base_model_name


def _run_domain(
    model_dir: Path, test_path: Path, output_path: Path | None,
    apply_rules: bool = False,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer_name = _load_model(model_dir)
    model.to(device)

    emotion_true, emotion_pred = [], []
    intent_true, intent_pred = [], []
    intensity_true, intensity_pred = [], []
    risk_true, risk_pred_binary = [], []

    if apply_rules:
        # Sample-by-sample inference so we have text + score dicts for rule application.
        from transformers import AutoTokenizer
        from backend.app.services.nlp_service import RealNLPService

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        EMOTION_IDX = {label: i for i, label in enumerate(EMOTION_LABELS)}
        INTENT_IDX = {label: i for i, label in enumerate(INTENT_LABELS)}

        records: list[dict] = []
        with open(test_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        with torch.no_grad():
            for rec in records:
                enc = tokenizer(
                    rec["text"], max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt",
                )
                out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))

                e_probs = torch.softmax(out.emotion_logits[0], dim=-1).cpu().tolist()
                i_probs = torch.softmax(out.intent_logits[0], dim=-1).cpu().tolist()
                intensity = float(out.intensity[0].item())
                risk_aux = float(out.risk_aux[0].item())

                emotion_scores = {k: round(v, 4) for k, v in zip(EMOTION_LABELS, e_probs)}
                intent_scores = {k: round(v, 4) for k, v in zip(INTENT_LABELS, i_probs)}

                pred_e = EMOTION_LABELS[int(torch.tensor(e_probs).argmax())]
                pred_i = INTENT_LABELS[int(torch.tensor(i_probs).argmax())]

                pred_e, pred_i = RealNLPService._apply_rules(
                    rec["text"], pred_e, pred_i, emotion_scores, intensity, intent_scores
                )

                emotion_pred.append(EMOTION_IDX[pred_e])
                intent_pred.append(INTENT_IDX[pred_i])
                intensity_pred.append(intensity)
                risk_pred_binary.append(int(risk_aux >= 0.5))

                emotion_true.append(EMOTION_IDX[rec["emotion_label"]])
                intent_true.append(INTENT_IDX[rec["intent_label"]])
                intensity_true.append(rec["intensity_score"])
                risk_true.append(int(rec["risk_flag"]))
    else:
        dataset = MentalHealthDataset(test_path, tokenizer_name)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        with torch.no_grad():
            for batch in loader:
                out = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                emotion_pred.extend(out.emotion_logits.argmax(-1).cpu().tolist())
                intent_pred.extend(out.intent_logits.argmax(-1).cpu().tolist())
                intensity_pred.extend(out.intensity.squeeze(-1).cpu().tolist())
                risk_pred_binary.extend((out.risk_aux.squeeze(-1) >= 0.5).long().cpu().tolist())

                emotion_true.extend(batch["emotion_label"].tolist())
                intent_true.extend(batch["intent_label"].tolist())
                intensity_true.extend(batch["intensity_score"].tolist())
                risk_true.extend(batch["risk_flag"].long().tolist())

    emotion_acc = accuracy_score(emotion_true, emotion_pred)
    emotion_macro_f1 = f1_score(emotion_true, emotion_pred, average="macro", zero_division=0)
    intent_acc = accuracy_score(intent_true, intent_pred)
    intent_macro_f1 = f1_score(intent_true, intent_pred, average="macro", zero_division=0)
    intensity_mae = sum(abs(p - t) for p, t in zip(intensity_pred, intensity_true)) / len(intensity_true)

    tp = sum(p == 1 and t == 1 for p, t in zip(risk_pred_binary, risk_true))
    fp = sum(p == 1 and t == 0 for p, t in zip(risk_pred_binary, risk_true))
    fn = sum(p == 0 and t == 1 for p, t in zip(risk_pred_binary, risk_true))
    risk_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    risk_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    risk_f1 = (
        2 * risk_precision * risk_recall / (risk_precision + risk_recall)
        if (risk_precision + risk_recall) > 0 else 0.0
    )

    results = {
        "mode": "domain",
        "emotion_accuracy": round(emotion_acc, 4),
        "emotion_macro_f1": round(emotion_macro_f1, 4),
        "intent_accuracy": round(intent_acc, 4),
        "intent_macro_f1": round(intent_macro_f1, 4),
        "intensity_mae": round(intensity_mae, 4),
        "risk_precision": round(risk_precision, 4),
        "risk_recall": round(risk_recall, 4),
        "risk_f1": round(risk_f1, 4),
    }

    # Print report
    print("\n=== Domain Evaluation ===")
    for k, v in results.items():
        if k != "mode":
            print(f"  {k:<22}: {v}")

    # Save confusion matrices
    em_cm = confusion_matrix(emotion_true, emotion_pred)
    _save_confusion(model_dir / "emotion_confusion.txt", em_cm, EMOTION_LABELS, "Emotion")
    in_cm = confusion_matrix(intent_true, intent_pred)
    _save_confusion(model_dir / "intent_confusion.txt", in_cm, INTENT_LABELS, "Intent")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def _run_public(model_dir: Path, test_path: Path, output_path: Path | None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer_name = _load_model(model_dir)
    model.to(device)

    dataset = PublicEmotionDataset(test_path, tokenizer_name)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    emotion_true, emotion_pred = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            emotion_pred.extend(out.emotion_logits.argmax(-1).cpu().tolist())
            emotion_true.extend(batch["emotion_label"].tolist())

    emotion_acc = accuracy_score(emotion_true, emotion_pred)
    emotion_macro_f1 = f1_score(emotion_true, emotion_pred, average="macro", zero_division=0)

    results = {
        "mode": "public",
        "emotion_accuracy": round(emotion_acc, 4),
        "emotion_macro_f1": round(emotion_macro_f1, 4),
        "intent_accuracy": "N/A",
        "intent_macro_f1": "N/A",
        "intensity_mae": "N/A",
        "risk_precision": "N/A",
        "risk_recall": "N/A",
        "risk_f1": "N/A",
    }

    print("\n=== Public Dataset Evaluation (emotion only) ===")
    print(f"  {'emotion_accuracy':<22}: {emotion_acc:.4f}")
    print(f"  {'emotion_macro_f1':<22}: {emotion_macro_f1:.4f}")
    print("  (intent / intensity / risk : N/A — public dataset, no labels)")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def _run_probe(model_dir: Path, test_path: Path, apply_rules: bool = False) -> dict:
    """Evaluate on the 30-sentence crisis probe set.

    Each probe record must have a ``tier`` field (0, 1, or 2) and the full
    annotation fields.  Tier 2 (crisis) failures are blocking.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer_name = _load_model(model_dir)
    model.to(device)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if apply_rules:
        from backend.app.services.nlp_service import RealNLPService

    records: list[dict] = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    tier_pass = {0: 0, 1: 0, 2: 0}
    tier_total = {0: 0, 1: 0, 2: 0}
    tier2_failures: list[str] = []

    INTENT_IDX = {label: i for i, label in enumerate(["venting", "seeking_advice",
                  "seeking_empathy", "crisis", "self_disclosure", "information_seeking"])}
    CRISIS_INTENT_IDX = INTENT_IDX["crisis"]

    with torch.no_grad():
        for rec in records:
            tier: int = rec["tier"]
            tier_total[tier] = tier_total.get(tier, 0) + 1

            enc = tokenizer(
                rec["text"],
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))

            pred_intent_idx = out.intent_logits.argmax(-1).item()
            pred_intensity = out.intensity.item()
            pred_risk_aux = out.risk_aux.item()

            if apply_rules:
                e_probs = torch.softmax(out.emotion_logits[0], dim=-1).cpu().tolist()
                i_probs = torch.softmax(out.intent_logits[0], dim=-1).cpu().tolist()
                emotion_scores = {k: round(v, 4) for k, v in zip(EMOTION_LABELS, e_probs)}
                intent_scores = {k: round(v, 4) for k, v in zip(INTENT_LABELS, i_probs)}
                pred_emotion = EMOTION_LABELS[int(torch.tensor(e_probs).argmax())]
                pred_intent = INTENT_LABELS[pred_intent_idx]
                pred_emotion, pred_intent = RealNLPService._apply_rules(
                    rec["text"], pred_emotion, pred_intent, emotion_scores, pred_intensity, intent_scores
                )
                pred_intent_idx = INTENT_IDX[pred_intent]

            if tier == 0:
                # Safe: risk_aux < 0.25 and not predicted as crisis
                passed = pred_risk_aux < 0.25 and pred_intent_idx != CRISIS_INTENT_IDX
            elif tier == 1:
                # Ambiguous: intensity in [0.30, 0.85], not crisis
                passed = pred_intensity >= 0.30 and pred_intensity <= 0.85 and pred_intent_idx != CRISIS_INTENT_IDX
            else:  # tier == 2
                # Crisis: must predict crisis intent AND risk_aux > 0.65
                passed = pred_intent_idx == CRISIS_INTENT_IDX and pred_risk_aux > 0.65
                if not passed:
                    tier2_failures.append(
                        f"  FAILED: \"{rec['text'][:50]}\" "
                        f"intent={INTENT_LABELS[pred_intent_idx]} "
                        f"risk_aux={pred_risk_aux:.3f}"
                    )

            if passed:
                tier_pass[tier] = tier_pass.get(tier, 0) + 1

    total_pass = sum(tier_pass.values())
    total = sum(tier_total.values())

    print("\n=== Crisis Probe Evaluation ===")
    tier_names = {0: "safe  ", 1: "ambig.", 2: "crisis"}
    for t in sorted(tier_total.keys()):
        marker = " ← must be 10/10" if t == 2 else ""
        print(f"  Tier {t} ({tier_names[t]}): {tier_pass.get(t, 0)}/{tier_total[t]} passed{marker}")
    print(f"  Overall        : {total_pass}/{total} passed")

    if tier2_failures:
        print("\n  BLOCKING — Tier 2 failures (crisis sentences not detected):")
        for msg in tier2_failures:
            print(msg)
        print("\n  Model must not be integrated into the backend until all Tier 2 sentences pass.")

    results = {
        "mode": "probe",
        "tier_0_pass": tier_pass.get(0, 0),
        "tier_0_total": tier_total.get(0, 0),
        "tier_1_pass": tier_pass.get(1, 0),
        "tier_1_total": tier_total.get(1, 0),
        "tier_2_pass": tier_pass.get(2, 0),
        "tier_2_total": tier_total.get(2, 0),
        "total_pass": total_pass,
        "total": total,
        "tier_2_failures": tier2_failures,
        "blocking": len(tier2_failures) > 0,
    }
    return results


def _save_confusion(path: Path, cm, labels: list[str], title: str) -> None:
    col_width = max(len(label) for label in labels) + 2
    lines = [f"{title} Confusion Matrix", ""]
    header = " " * col_width + "".join(f"{label:>{col_width}}" for label in labels)
    lines.append(header)
    for i, row in enumerate(cm):
        row_str = f"{labels[i]:<{col_width}}" + "".join(f"{v:>{col_width}}" for v in row)
        lines.append(row_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Confusion matrix saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mental-health NLP model")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--test", required=True, help="Path to test JSONL file")
    parser.add_argument("--mode", choices=["domain", "public", "probe"], default="domain")
    parser.add_argument("--output", default=None, help="Optional path to save JSON results")
    parser.add_argument(
        "--apply-rules", action="store_true",
        help="Apply RealNLPService post-processing rules after model inference",
    )
    args = parser.parse_args()

    model_dir = Path(args.model)
    test_path = Path(args.test)
    output_path = Path(args.output) if args.output else None

    if args.mode == "domain":
        _run_domain(model_dir, test_path, output_path, apply_rules=args.apply_rules)
    elif args.mode == "public":
        _run_public(model_dir, test_path, output_path)
    else:
        _run_probe(model_dir, test_path, apply_rules=args.apply_rules)


if __name__ == "__main__":
    main()
