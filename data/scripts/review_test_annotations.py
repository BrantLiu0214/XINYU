from __future__ import annotations

import json
from pathlib import Path


FOCUS_PATTERNS = [
    "活着",
    "没意思",
    "没意义",
    "没什么劲",
    "没劲",
    "无所谓",
    "不用醒",
    "醒不过来",
    "消失",
    "楼下",
    "阳台",
    "马路",
    "提不起劲",
    "发呆",
    "耗着",
    "熬着",
    "就这样吧",
    "没人会发现",
    "没人会在意",
]


def main() -> None:
    path = Path("data/processed/mental_dialogue_test.jsonl")
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    focus = [r for r in recs if any(p in r["text"] for p in FOCUS_PATTERNS)]
    print(f"focus_count={len(focus)}")
    for rec in focus:
        print(
            f'{rec["id"]}\t{rec["emotion_label"]}\t{rec["intent_label"]}\t'
            f'{rec["intensity_score"]}\t{rec["risk_flag"]}\t{rec["text"]}'
        )


if __name__ == "__main__":
    main()
