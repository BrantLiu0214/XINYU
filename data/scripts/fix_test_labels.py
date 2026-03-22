"""Apply audit corrections to mental_dialogue_test.jsonl.
Corrections sourced from Claude Opus 4.6 full audit of 500 records.

Run from project root:
    python data/scripts/fix_test_labels.py
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

PATH = Path("data/processed/mental_dialogue_test.jsonl")

# ── emotion corrections {1-based line: new_label} ─────────────────────────────
EMOTION_FIXES: dict[int, str] = {
    15: "anxiety",
    16: "hopelessness",
    25: "shame",
    36: "sadness",
    37: "fear",
    41: "neutral",
    44: "sadness",
    54: "neutral",
    59: "hopelessness",
    63: "neutral",
    65: "hopelessness",
    76: "neutral",
    78: "hopelessness",
    84: "hopelessness",
    90: "hopelessness",
    96: "neutral",
    99: "hopelessness",
    101: "hopelessness",
    103: "hopelessness",
    104: "sadness",
    105: "hopelessness",
    106: "hopelessness",
    107: "anxiety",
    108: "shame",
    111: "shame",
    112: "neutral",
    117: "hopelessness",
    118: "anxiety",
    129: "shame",
    131: "neutral",
    136: "hopelessness",
    137: "hopelessness",
    138: "hopelessness",
    141: "hopelessness",
    142: "hopelessness",
    147: "hopelessness",
    148: "shame",
    149: "sadness",
    150: "hopelessness",
    152: "hopelessness",
    153: "hopelessness",
    160: "anxiety",
    161: "neutral",
    164: "anxiety",
    168: "hopelessness",
    171: "neutral",
    174: "anger",
    181: "anxiety",
    182: "hopelessness",
    185: "anxiety",
    186: "hopelessness",
    187: "shame",
    191: "neutral",
    192: "hopelessness",
    200: "anxiety",
    201: "shame",
    203: "hopelessness",
    206: "hopelessness",
    211: "hopelessness",
    213: "shame",
    219: "neutral",
    222: "neutral",
    225: "hopelessness",
    237: "sadness",
    239: "anxiety",
    246: "hopelessness",
    249: "shame",
    251: "neutral",
    253: "hopelessness",
    256: "hopelessness",
    258: "sadness",
    260: "shame",
    263: "neutral",
    269: "neutral",
    272: "neutral",
    275: "hopelessness",
    276: "hopelessness",
    284: "shame",
    285: "hopelessness",
    291: "anxiety",
    293: "shame",
    295: "sadness",
    297: "hopelessness",
    302: "neutral",
    306: "hopelessness",
    310: "hopelessness",
    312: "anxiety",
    316: "hopelessness",
    318: "hopelessness",
    320: "anxiety",
    322: "sadness",
    325: "sadness",
    327: "hopelessness",
    329: "anxiety",
    330: "neutral",
    333: "hopelessness",
    335: "hopelessness",
    337: "neutral",
    340: "shame",
    342: "hopelessness",
    344: "anxiety",
    345: "neutral",
    348: "hopelessness",
    349: "anxiety",
    350: "neutral",
    354: "neutral",
    355: "hopelessness",
    356: "shame",
    357: "hopelessness",
    358: "neutral",
    359: "shame",
    360: "hopelessness",
    361: "hopelessness",
    362: "neutral",
    367: "hopelessness",
    368: "hopelessness",
    373: "shame",
    374: "neutral",
    377: "anxiety",
    380: "hopelessness",
    381: "anxiety",
    383: "hopelessness",
    389: "sadness",
    390: "hopelessness",
    392: "anxiety",
    393: "hopelessness",
    395: "shame",
    396: "anxiety",
    399: "sadness",
    405: "hopelessness",
    407: "hopelessness",
    411: "hopelessness",
    412: "hopelessness",
    413: "hopelessness",
    417: "anxiety",
    418: "hopelessness",
    424: "neutral",
    429: "shame",
    437: "shame",
    438: "neutral",
    440: "hopelessness",
    442: "shame",
    445: "hopelessness",
    450: "neutral",
    451: "anxiety",
    452: "hopelessness",
    453: "neutral",
    464: "shame",
    467: "neutral",
    476: "neutral",
    483: "shame",
    496: "neutral",
    497: "neutral",
    499: "anxiety",
}

# ── intent corrections {1-based line: new_label} ──────────────────────────────
INTENT_FIXES: dict[int, str] = {
    4:   "self_disclosure",
    6:   "venting",
    12:  "venting",
    13:  "venting",
    14:  "venting",
    21:  "self_disclosure",
    32:  "self_disclosure",
    39:  "self_disclosure",
    42:  "self_disclosure",
    47:  "self_disclosure",
    51:  "venting",
    54:  "self_disclosure",   # was patched to self_disclosure earlier; stays
    63:  "self_disclosure",
    64:  "venting",
    68:  "self_disclosure",
    86:  "self_disclosure",
    88:  "seeking_empathy",
    96:  "self_disclosure",
    97:  "venting",
    112: "self_disclosure",
    113: "self_disclosure",
    129: "self_disclosure",
    131: "self_disclosure",
    132: "self_disclosure",
    144: "self_disclosure",
    163: "venting",
    178: "venting",
    179: "venting",
    230: "self_disclosure",
    255: "self_disclosure",
    288: "self_disclosure",
    328: "venting",
    332: "venting",
    358: "seeking_empathy",
    385: "seeking_empathy",
    387: "venting",
    397: "venting",
    407: "crisis",
    434: "self_disclosure",
    444: "venting",
    447: "self_disclosure",
    455: "venting",
    457: "venting",
    458: "seeking_advice",
    459: "venting",
    464: "venting",
    470: "venting",
    471: "venting",
    474: "venting",
    480: "self_disclosure",
    481: "crisis",
    491: "venting",
    498: "venting",
}

# Lines being demoted FROM crisis (risk_flag → False)
DEMOTE_CRISIS = {6, 12, 13, 14, 51, 64, 97, 178, 179, 328, 397, 455, 459, 470, 471, 474, 491}
# Lines being promoted TO crisis (risk_flag → True, intensity ≥ 0.80)
PROMOTE_CRISIS = {407, 481}


def swap_scores(scores: dict, label: str) -> dict:
    top = max(scores, key=lambda k: scores[k])
    if top == label:
        return scores
    scores = dict(scores)
    scores[label], scores[top] = scores[top], scores[label]
    total = sum(scores.values())
    if abs(total - 1.0) > 1e-9:
        scores[label] = round(scores[label] + (1.0 - total), 4)
    return scores


def main() -> None:
    shutil.copy(PATH, PATH.with_suffix(".jsonl.bak2"))

    records = []
    with open(PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    emotion_changed = 0
    intent_changed = 0
    risk_changed = 0

    for i, r in enumerate(records):
        ln = i + 1

        # Emotion correction
        if ln in EMOTION_FIXES:
            new_e = EMOTION_FIXES[ln]
            if r["emotion_label"] != new_e:
                r["emotion_scores"] = swap_scores(r["emotion_scores"], new_e)
                r["emotion_label"] = new_e
                emotion_changed += 1

        # Intent correction
        if ln in INTENT_FIXES:
            new_i = INTENT_FIXES[ln]
            if r["intent_label"] != new_i:
                r["intent_scores"] = swap_scores(r["intent_scores"], new_i)
                r["intent_label"] = new_i
                intent_changed += 1

        # Risk flag adjustments
        if ln in DEMOTE_CRISIS and r["risk_flag"]:
            r["risk_flag"] = False
            risk_changed += 1
        elif ln in PROMOTE_CRISIS:
            if not r["risk_flag"]:
                r["risk_flag"] = True
                risk_changed += 1
            if r["intensity_score"] < 0.80:
                r["intensity_score"] = 0.80

    with open(PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Emotion labels changed : {emotion_changed}")
    print(f"Intent labels changed  : {intent_changed}")
    print(f"Risk flags changed     : {risk_changed}")
    print(f"Backup: {PATH.with_suffix('.jsonl.bak2')}")

    # Final distribution
    from collections import Counter
    emotion_counts: Counter = Counter()
    intent_counts: Counter = Counter()
    risk_true = 0
    for r in records:
        emotion_counts[r["emotion_label"]] += 1
        intent_counts[r["intent_label"]] += 1
        if r["risk_flag"]:
            risk_true += 1
    n = len(records)
    print("\nFinal emotion distribution:")
    for lbl, cnt in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        print("  %-20s %4d  (%.1f%%)" % (lbl, cnt, cnt / n * 100))
    print("\nFinal intent distribution:")
    for lbl, cnt in sorted(intent_counts.items(), key=lambda x: -x[1]):
        print("  %-25s %4d  (%.1f%%)" % (lbl, cnt, cnt / n * 100))
    print("\nrisk_flag=True: %d (%.1f%%)" % (risk_true, risk_true / n * 100))


if __name__ == "__main__":
    main()
