"""Convert public Chinese emotion datasets to the annotation-spec JSONL format.

Supports:
  weibo     — NLPCC 2018 Weibo Emotion task 1 (tab-separated, 6 emotion classes)
              or weibo2018-main binary sentiment (comma-separated, 0/1 labels)
  nlpcc2014 — NLPCC 2014 Weibo Emotion Classification XML (7 emotion classes)
  cped      — CPED via HuggingFace datasets (12 emotion classes)

The output files carry only emotion_label and emotion_scores (the public
datasets do not provide intent, intensity, or risk labels).  These files are
used exclusively for cross-dataset emotion evaluation and must not be used for
training.

Usage:
    # Weibo (manually downloaded):
    python data/scripts/convert_public_dataset.py \\
        --dataset weibo \\
        --input data/raw/weibo_emotion_train.txt \\
        --output data/processed/weibo_emotion_eval.jsonl

    # NLPCC 2014 (manually downloaded XML):
    python data/scripts/convert_public_dataset.py \\
        --dataset nlpcc2014 \\
        --input "data/raw/NLPCC2014/Training data for Emotion Classification.xml" \\
        --output data/processed/nlpcc2014_emotion_eval.jsonl

    # CPED via HuggingFace:
    python data/scripts/convert_public_dataset.py \\
        --dataset cped \\
        --input hf \\
        --output data/processed/cped_emotion_eval.jsonl

Notes on the Weibo dataset:
  Download from NLPCC 2018 Task 1:
    http://tcci.ccf.org.cn/conference/2018/taskdata.php
  The file format is typically tab-separated: <label>\\t<text>
  where label is one of: happiness, sadness, disgust, anger, fear, surprise.

Notes on CPED:
  Dataset identifier on HuggingFace: "silver-yibu/CPED"
  Use --input hf to trigger automatic download.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Label mappings ────────────────────────────────────────────────────────────

# These mappings are locked before evaluation begins.  Do not change them
# after running evaluate.py — that would invalidate the comparison.

# NLPCC 2014 emotion-type labels → our taxonomy
NLPCC2014_TO_OURS: dict[str, str] = {
    "none":      "neutral",   # no emotion
    "like":      "neutral",   # positive affect; no positive label in our taxonomy
    "happiness": "neutral",   # positive affect
    "disgust":   "anger",     # closest negative affect
    "sadness":   "sadness",
    "anger":     "anger",
    "surprise":  "neutral",   # ambiguous valence
    "fear":      "fear",
}

WEIBO_TO_OURS: dict[str, str] = {
    "happiness": "neutral",    # positive affect; no positive label in our taxonomy
    "sadness": "sadness",
    "disgust": "anger",        # closest negative affect
    "anger": "anger",
    "fear": "fear",
    "surprise": "neutral",     # ambiguous valence
}

CPED_TO_OURS: dict[str, str] = {
    "anxious": "anxiety",
    "sad": "sadness",
    "disappointed": "sadness",
    "angry": "anger",
    "scared": "fear",
    "fearful": "fear",
    "embarrassed": "shame",
    "neutral": "neutral",
    "satisfied": "neutral",
    "excited": "neutral",
    "surprised": "neutral",
    # All other CPED labels not listed here default to "neutral"
}

OUR_EMOTION_LABELS = ["anxiety", "sadness", "anger", "fear", "shame", "hopelessness", "neutral"]


def _make_scores(label: str) -> dict[str, float]:
    """Create emotion_scores with 1.0 on the given label and 0.0 on all others."""
    return {k: (1.0 if k == label else 0.0) for k in OUR_EMOTION_LABELS}


# ── Weibo converter ───────────────────────────────────────────────────────────

def convert_weibo(input_path: Path, output_path: Path) -> None:
    """Convert Weibo sentiment corpus.

    Supports two formats auto-detected from the first data line:

    Format A — NLPCC 2018 6-class emotion, tab-separated:
        <label>\\t<text>   or   <id>\\t<label>\\t<text>
        where label ∈ {happiness, sadness, disgust, anger, fear, surprise}

    Format B — Binary sentiment, comma-separated (weibo2018-main):
        <id>,<polarity>,<text>
        where polarity ∈ {0=negative, 1=positive}

    For Format B the polarity is mapped to our taxonomy as follows:
        1 (positive) → neutral
        0 (negative) → sadness  (placeholder; evaluation is polarity accuracy)
    The output records carry a ``polarity`` field (0/1) so that evaluate.py
    can compute binary polarity accuracy instead of 7-class emotion accuracy
    when running in public mode on this dataset.
    """
    records: list[dict] = []
    skipped = 0
    label_counts: dict[str, int] = {}

    with open(input_path, encoding="utf-8", errors="replace") as f:
        lines = [l.strip() for l in f if l.strip()]

    # Auto-detect format from first line
    first = lines[0] if lines else ""
    # Format B: comma-separated with numeric second field
    use_binary = "," in first and not "\t" in first

    for line in lines:
        if use_binary:
            parts = line.split(",", 2)
            if len(parts) < 3:
                skipped += 1
                continue
            raw_label = parts[1].strip()
            text = parts[2].strip()
            if raw_label not in ("0", "1"):
                skipped += 1
                continue
            polarity = int(raw_label)
            our_label = "neutral" if polarity == 1 else "sadness"
            label_counts[raw_label] = label_counts.get(raw_label, 0) + 1
            records.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "emotion_label": our_label,
                "emotion_scores": _make_scores(our_label),
                "polarity": polarity,          # 1=positive, 0=negative
                "source": "weibo2018_binary",
                "source_label": raw_label,
            })
        else:
            parts = line.split("\t")
            if len(parts) < 2:
                skipped += 1
                continue
            raw_label = parts[0].strip().lower()
            text = parts[-1].strip()
            if raw_label not in WEIBO_TO_OURS and len(parts) >= 3:
                raw_label = parts[1].strip().lower()
                text = parts[-1].strip()
            if raw_label not in WEIBO_TO_OURS:
                skipped += 1
                continue
            our_label = WEIBO_TO_OURS[raw_label]
            label_counts[raw_label] = label_counts.get(raw_label, 0) + 1
            records.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "emotion_label": our_label,
                "emotion_scores": _make_scores(our_label),
                "source": "weibo_nlpcc2018",
                "source_label": raw_label,
            })

    fmt = "binary-sentiment" if use_binary else "6-class-emotion"
    print(f"  Detected format: {fmt}")
    _write_and_report(records, output_path, "Weibo", skipped, label_counts)


# ── NLPCC 2014 converter ──────────────────────────────────────────────────────

def convert_nlpcc2014(input_path: Path, output_path: Path) -> None:
    """Convert NLPCC 2014 Weibo Emotion Classification XML.

    Format: XML with <weibo id="N" emotion-type1="..." emotion-type2="...">
    containing one or more <sentence> children.  The primary label is
    emotion-type1.  All sentences within a weibo are concatenated as the text.
    Records with emotion-type1="none" are included and mapped to neutral.
    """
    with open(input_path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    # Fix non-standard XML declarations (e.g. extra attributes)
    import re
    content = re.sub(r"<\?xml[^?]*\?>", '<?xml version="1.0" encoding="UTF-8"?>', content)

    root = ET.fromstring(content)
    records: list[dict] = []
    skipped = 0
    label_counts: dict[str, int] = {}

    for weibo in root.findall("weibo"):
        raw_label = weibo.attrib.get("emotion-type1", "").lower().strip()
        if raw_label not in NLPCC2014_TO_OURS:
            skipped += 1
            continue

        sentences = [s.text.strip() for s in weibo.findall("sentence") if s.text and s.text.strip()]
        if not sentences:
            skipped += 1
            continue
        text = "".join(sentences)

        our_label = NLPCC2014_TO_OURS[raw_label]
        label_counts[raw_label] = label_counts.get(raw_label, 0) + 1
        records.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "emotion_label": our_label,
            "emotion_scores": _make_scores(our_label),
            "source": "nlpcc2014_weibo",
            "source_label": raw_label,
        })

    _write_and_report(records, output_path, "NLPCC2014", skipped, label_counts)


# ── CPED converter ────────────────────────────────────────────────────────────

def convert_cped(output_path: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required for CPED conversion.  "
              "Install with: pip install datasets")
        sys.exit(1)

    print("Downloading CPED from HuggingFace…")
    try:
        ds = load_dataset("silver-yibu/CPED", trust_remote_code=True)
    except Exception as exc:
        print(f"ERROR loading CPED: {exc}")
        print("If the dataset ID has changed, update convert_public_dataset.py accordingly.")
        sys.exit(1)

    records: list[dict] = []
    skipped = 0
    label_counts: dict[str, int] = {}

    # CPED may have 'train', 'validation', 'test' splits; use all for evaluation corpus
    for split_name in ds:
        split = ds[split_name]
        for sample in split:
            # CPED field names may vary; try common ones
            text = sample.get("utterance") or sample.get("text") or sample.get("content", "")
            raw_label = (sample.get("emotion") or sample.get("emotion_label") or "").lower()
            if not text or not raw_label:
                skipped += 1
                continue

            our_label = CPED_TO_OURS.get(raw_label, "neutral")
            label_counts[raw_label] = label_counts.get(raw_label, 0) + 1

            records.append({
                "id": str(uuid.uuid4()),
                "text": str(text).strip(),
                "emotion_label": our_label,
                "emotion_scores": _make_scores(our_label),
                "source": "cped_huggingface",
                "source_label": raw_label,
            })

    _write_and_report(records, output_path, "CPED", skipped, label_counts)


def _write_and_report(
    records: list[dict],
    output_path: Path,
    dataset_name: str,
    skipped: int,
    label_counts: dict[str, int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(records)
    print(f"\n[{dataset_name}] {total} samples written to {output_path}  ({skipped} skipped)")
    print("  Source label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        mapped = WEIBO_TO_OURS.get(label) or NLPCC2014_TO_OURS.get(label) or CPED_TO_OURS.get(label, "neutral")
        print(f"    {label:<20} → {mapped:<14} : {count} ({count/total:.1%})")

    print("  Our taxonomy distribution (after mapping):")
    our_counts: dict[str, int] = {}
    for rec in records:
        lbl = rec["emotion_label"]
        our_counts[lbl] = our_counts.get(lbl, 0) + 1
    for label in OUR_EMOTION_LABELS:
        cnt = our_counts.get(label, 0)
        print(f"    {label:<16}: {cnt} ({cnt/total:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert public Chinese emotion datasets")
    parser.add_argument("--dataset", choices=["weibo", "nlpcc2014", "cped"], required=True)
    parser.add_argument("--input", required=True,
                        help="Path to raw dataset file, or 'hf' to download from HuggingFace")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.dataset == "weibo":
        if args.input.lower() == "hf":
            print("ERROR: Weibo dataset must be downloaded manually.")
            sys.exit(1)
        convert_weibo(Path(args.input), output_path)
    elif args.dataset == "nlpcc2014":
        if args.input.lower() == "hf":
            print("ERROR: NLPCC2014 dataset must be downloaded manually.")
            sys.exit(1)
        convert_nlpcc2014(Path(args.input), output_path)
    else:  # cped
        convert_cped(output_path)


if __name__ == "__main__":
    main()
