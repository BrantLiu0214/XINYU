"""Generate the thesis comparison Markdown table from four evaluation JSON files.

Usage:
    python nlp/eval/compare_models.py \\
        --roberta-domain-results nlp/artifacts/roberta-domain/domain_results.json \\
        --bert-domain-results    nlp/artifacts/bert-domain/domain_results.json \\
        --roberta-public-results nlp/artifacts/roberta-domain/public_results.json \\
        --bert-public-results    nlp/artifacts/bert-domain/public_results.json \\
        --output docs/experiment-results/nlp_comparison.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_table(rd: dict, bd: dict, rp: dict, bp: dict) -> str:
    rows = [
        ("Emotion Accuracy",  "emotion_accuracy"),
        ("Emotion Macro-F1",  "emotion_macro_f1"),
        ("Intent Accuracy",   "intent_accuracy"),
        ("Intent Macro-F1",   "intent_macro_f1"),
        ("Intensity MAE",     "intensity_mae"),
        ("Risk Precision",    "risk_precision"),
        ("Risk Recall",       "risk_recall"),
        ("Risk F1",           "risk_f1"),
    ]

    header = (
        "| Metric "
        "| RoBERTa-wwm-ext (domain) "
        "| BERT-base (domain) "
        "| RoBERTa-wwm-ext (public) "
        "| BERT-base (public) |"
    )
    sep = "|--------|--------------------------|--------------------|--------------------------|--------------------|"

    lines = [
        "# NLP Model Comparison",
        "",
        "RoBERTa-wwm-ext (`hfl/chinese-roberta-wwm-ext`) vs. BERT-base (`bert-base-chinese`).",
        "Both models share the same multi-task architecture and training hyperparameters.",
        "Public dataset evaluation covers emotion classification only (no intent/intensity/risk labels available).",
        "",
        header,
        sep,
    ]

    for display, key in rows:
        r_dom = _fmt(rd.get(key, "N/A"))
        b_dom = _fmt(bd.get(key, "N/A"))
        r_pub = _fmt(rp.get(key, "N/A"))
        b_pub = _fmt(bp.get(key, "N/A"))
        lines.append(f"| {display} | {r_dom} | {b_dom} | {r_pub} | {b_pub} |")

    lines += [
        "",
        "**Notes**",
        "- Domain dataset: self-constructed, human-annotated test split (500 samples).",
        "- Public dataset: NLPCC 2014 Weibo Emotion (7-class, mapped to our taxonomy; anxiety/shame/hopelessness absent).",
        "- N/A indicates the metric is unavailable for that dataset due to absent labels.",
        "- Risk Recall target: ≥ 0.90 on domain dataset.",
        "- Emotion Accuracy target: ≥ 0.80 on domain dataset.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model comparison Markdown table")
    parser.add_argument("--roberta-domain-results", required=True, dest="rd")
    parser.add_argument("--bert-domain-results", required=True, dest="bd")
    parser.add_argument("--roberta-public-results", required=True, dest="rp")
    parser.add_argument("--bert-public-results", required=True, dest="bp")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rd = _load(Path(args.rd))
    bd = _load(Path(args.bd))
    rp = _load(Path(args.rp))
    bp = _load(Path(args.bp))

    table = build_table(rd, bd, rp, bp)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table, encoding="utf-8")
    print(f"Comparison table written to {out}")


if __name__ == "__main__":
    main()
