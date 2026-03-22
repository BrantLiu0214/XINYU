"""Inter-annotator agreement (Cohen's κ) for emotion and intent labels.

Usage:
    python nlp/eval/compute_iaa.py \\
        --annotator-a data/processed/iaa_annotator_a.jsonl \\
        --annotator-b data/processed/iaa_annotator_b.jsonl

Both files must contain the same sample IDs in any order.  Only samples
present in both files are included in the κ calculation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from sklearn.metrics import cohen_kappa_score
except ModuleNotFoundError:
    def cohen_kappa_score(labels_a, labels_b):
        if len(labels_a) != len(labels_b):
            raise ValueError("label sequences must have the same length")
        n = len(labels_a)
        if n == 0:
            raise ValueError("label sequences must be non-empty")

        observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n

        counts_a = Counter(labels_a)
        counts_b = Counter(labels_b)
        expected = sum(
            (counts_a[label] / n) * (counts_b[label] / n)
            for label in set(counts_a) | set(counts_b)
        )

        if expected == 1.0:
            return 1.0
        return (observed - expected) / (1 - expected)


def _load_by_id(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[rec["id"]] = rec
    return records


def compute_kappa(path_a: Path, path_b: Path, output_path: Path | None) -> dict:
    a_records = _load_by_id(path_a)
    b_records = _load_by_id(path_b)

    common_ids = sorted(set(a_records) & set(b_records))
    if not common_ids:
        print("ERROR: No matching sample IDs found between the two annotation files.")
        sys.exit(1)

    missing_a = set(b_records) - set(a_records)
    missing_b = set(a_records) - set(b_records)
    if missing_a or missing_b:
        print(f"Warning: {len(missing_a)} IDs in B but not A; {len(missing_b)} in A but not B. "
              f"Using {len(common_ids)} shared samples.")

    emotion_a = [a_records[sid]["emotion_label"] for sid in common_ids]
    emotion_b = [b_records[sid]["emotion_label"] for sid in common_ids]
    intent_a = [a_records[sid]["intent_label"] for sid in common_ids]
    intent_b = [b_records[sid]["intent_label"] for sid in common_ids]

    emotion_kappa = cohen_kappa_score(emotion_a, emotion_b)
    intent_kappa = cohen_kappa_score(intent_a, intent_b)

    results = {
        "n_samples": len(common_ids),
        "emotion_kappa": round(float(emotion_kappa), 4),
        "intent_kappa": round(float(intent_kappa), 4),
        "emotion_kappa_target": 0.70,
        "intent_kappa_target": 0.70,
        "emotion_kappa_pass": emotion_kappa >= 0.70,
        "intent_kappa_pass": intent_kappa >= 0.70,
    }

    print(f"\n=== Inter-Annotator Agreement (n={len(common_ids)}) ===")
    status_e = "PASS" if results["emotion_kappa_pass"] else "FAIL — reconciliation required"
    status_i = "PASS" if results["intent_kappa_pass"] else "FAIL — reconciliation required"
    print(f"  emotion_kappa : {emotion_kappa:.4f}  [{status_e}]  (target ≥ 0.70)")
    print(f"  intent_kappa  : {intent_kappa:.4f}  [{status_i}]  (target ≥ 0.70)")

    # List disagreements to help with reconciliation if needed
    if not results["emotion_kappa_pass"] or not results["intent_kappa_pass"]:
        print("\n  Disagreeing samples (first 20):")
        count = 0
        for sid in common_ids:
            e_diff = a_records[sid]["emotion_label"] != b_records[sid]["emotion_label"]
            i_diff = a_records[sid]["intent_label"] != b_records[sid]["intent_label"]
            if e_diff or i_diff:
                print(f"    {sid}: emotion {a_records[sid]['emotion_label']} vs "
                      f"{b_records[sid]['emotion_label']}, "
                      f"intent {a_records[sid]['intent_label']} vs "
                      f"{b_records[sid]['intent_label']}")
                count += 1
                if count >= 20:
                    remaining = sum(
                        1 for s in common_ids
                        if a_records[s]["emotion_label"] != b_records[s]["emotion_label"]
                        or a_records[s]["intent_label"] != b_records[s]["intent_label"]
                    ) - 20
                    if remaining > 0:
                        print(f"    ... and {remaining} more.")
                    break

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  Results saved to {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute inter-annotator agreement")
    parser.add_argument("--annotator-a", required=True, dest="annotator_a")
    parser.add_argument("--annotator-b", required=True, dest="annotator_b")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    compute_kappa(Path(args.annotator_a), Path(args.annotator_b),
                  Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
