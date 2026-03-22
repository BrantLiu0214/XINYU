from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


EMOTIONS = ["anxiety", "sadness", "anger", "fear", "shame", "hopelessness", "neutral"]
INTENTS = [
    "venting",
    "seeking_advice",
    "seeking_empathy",
    "crisis",
    "self_disclosure",
    "information_seeking",
]

OVERRIDES: dict[str, dict[str, object]] = {
    "5cef9fe0-ce34-4ad4-8108-fdca7bac383a": {
        "emotion_label": "hopelessness",
        "intent_label": "crisis",
        "intensity_score": 0.86,
        "risk_flag": True,
    },
    "9037361c-e663-46db-b28d-9df49cab50c9": {
        "emotion_label": "hopelessness",
        "intent_label": "crisis",
        "intensity_score": 0.9,
        "risk_flag": True,
    },
    "c59dfec8-1194-4567-ab84-9f0c93813b69": {
        "emotion_label": "hopelessness",
        "intent_label": "crisis",
        "intensity_score": 0.9,
        "risk_flag": True,
    },
    "e5c21a4c-f46a-4248-8740-986a5822d49f": {
        "emotion_label": "hopelessness",
        "intent_label": "crisis",
        "intensity_score": 0.9,
        "risk_flag": True,
    },
    "89066d83-5bcd-4bf8-ba37-051c7477d854": {
        "emotion_label": "hopelessness",
        "intent_label": "crisis",
        "intensity_score": 0.86,
        "risk_flag": True,
    },
    "fcbefdd9-6940-4c12-bc31-86fea0681dc6": {
        "emotion_label": "hopelessness",
        "intent_label": "crisis",
        "intensity_score": 0.93,
        "risk_flag": True,
    },
    "c4954a44-ddf8-42ef-93da-37c95a887517": {
        "emotion_label": "hopelessness",
        "intent_label": "crisis",
        "intensity_score": 0.88,
        "risk_flag": True,
    },
    "fe1c2e6c-2524-4ba6-9a82-191f006790c2": {
        "emotion_label": "hopelessness",
        "intent_label": "crisis",
        "intensity_score": 0.9,
        "risk_flag": True,
    },
    "998093e9-8ec1-4951-8c81-23c12ef9890e": {
        "emotion_label": "hopelessness",
        "intent_label": "crisis",
        "intensity_score": 0.88,
        "risk_flag": True,
    },
    "2606bc57-70d1-42eb-830e-7fa29d155f35": {
        "emotion_label": "hopelessness",
        "intent_label": "crisis",
        "intensity_score": 0.88,
        "risk_flag": True,
    },
    "9e3d0c1f-017e-4cff-9198-90b7393e9465": {
        "emotion_label": "hopelessness",
        "intent_label": "venting",
        "intensity_score": 0.76,
        "risk_flag": False,
    },
    "b5468c92-4ff7-4af6-bff7-2410303dae47": {
        "emotion_label": "hopelessness",
        "intent_label": "venting",
        "intensity_score": 0.76,
        "risk_flag": False,
    },
    "9705fb0e-803d-45a8-9567-8b7f20e6fff6": {
        "emotion_label": "hopelessness",
        "intent_label": "venting",
        "intensity_score": 0.78,
        "risk_flag": False,
    },
    "fef6f860-7645-4742-ab42-7abf5dae033a": {
        "emotion_label": "hopelessness",
        "intent_label": "venting",
        "intensity_score": 0.76,
        "risk_flag": False,
    },
}


def has_any(text: str, items: list[str]) -> bool:
    return any(item in text for item in items)


def make_scores(labels: list[str], primary: str, secondary: str) -> dict[str, float]:
    scores = {label: 0.0 for label in labels}
    scores[primary] = 0.8
    scores[secondary] = 0.15
    leftover = "neutral" if labels is EMOTIONS and primary != "neutral" and secondary != "neutral" else (
        "self_disclosure" if labels is INTENTS and primary == "venting" and secondary != "self_disclosure" else (
            "venting" if labels is INTENTS and primary != "venting" and secondary != "venting" else labels[0]
        )
    )
    scores[leftover] += 0.05
    diff = round(1.0 - sum(scores.values()), 4)
    scores[primary] = round(scores[primary] + diff, 4)
    return {k: round(v, 4) for k, v in scores.items()}


def classify(text: str) -> dict:
    crisis = False
    if has_any(
        text,
        [
            "不想活",
            "不用醒",
            "醒不过来",
            "消失了",
            "消失",
            "没我大家都会轻松",
            "没我大家都轻松",
            "做什么都没意义",
            "没什么意义",
            "没意义了",
        ],
    ):
        crisis = True
    if "活着" in text and has_any(
        text,
        [
            "没什么意思",
            "没意思",
            "没什么劲",
            "没劲",
            "没意义",
            "无所谓",
            "没滋味",
            "没奔头",
            "没什么好期待",
            "没什么特别的意思",
        ],
    ):
        crisis = True
    if has_any(text, ["要是一觉睡下去不用醒", "睡下去不用醒", "再也醒不过来"]):
        crisis = True
    if has_any(
        text,
        [
            "醒不来也无所谓",
            "睡着醒不来也无所谓",
            "就这样算了也挺好的",
            "要是掉下去是不是就轻松了",
        ],
    ):
        crisis = True
    if has_any(text, ["看着楼下", "盯着阳台", "阳台就发好久的呆", "站阳台"]) and has_any(
        text, ["没意思", "没意义", "轻松点", "消失"]
    ):
        crisis = True
    if "马路上" in text and "没注意" in text and has_any(text, ["不想动", "没意思", "无所谓"]):
        crisis = True

    if crisis:
        intent = "crisis"
    elif has_any(text, ["吃药", "药"]) and has_any(text, ["该不该", "要不要", "怎么", "能不能"]) and has_any(
        text, ["调身体", "坚持", "吃"]
    ):
        intent = "information_seeking"
    elif has_any(text, ["怎么办", "该不该", "要不要", "什么时候才能", "怎么缓过来", "怎么撑", "好不好", "能不能"]):
        intent = "seeking_advice"
    elif has_any(text, ["对吧", "我太矫情", "我太没用", "是我太", "也没处说", "没人真的在乎"]):
        intent = "seeking_empathy"
    elif has_any(
        text,
        [
            "我一个人带孩子",
            "单亲",
            "读了三年",
            "吃了三年药",
            "吃了五年药",
            "快三年了",
            "这么多年",
            "本来就",
        ],
    ) and not has_any(
        text,
        [
            "睡不着",
            "失眠",
            "没胃口",
            "吃不下",
            "头疼",
            "喘不上气",
            "胸口闷",
            "撑不住",
            "熬不住",
            "扛不住",
            "没什么意思",
            "没意义",
            "不用醒",
            "消失",
            "哭",
            "眼泪",
            "空落落",
            "提不起劲",
            "发呆",
        ],
    ):
        intent = "self_disclosure"
    else:
        intent = "venting"

    if crisis:
        emotion = "hopelessness"
    elif has_any(text, ["没用", "矫情", "不该", "拖累", "对不起", "不懂事", "添负担", "不该奢求"]):
        emotion = "shame"
    elif has_any(text, ["害怕", "心慌", "不敢", "担心", "总怕"]):
        emotion = "fear"
    elif has_any(
        text,
        [
            "睡不着",
            "失眠",
            "没胃口",
            "吃不下",
            "不想吃饭",
            "头疼",
            "头一直",
            "提不上气",
            "喘不上气",
            "胸口闷",
            "胸口堵",
            "撑不住",
            "熬不住",
            "扛不住",
            "压得",
            "发胀疼",
            "坐立难安",
        ],
    ):
        emotion = "anxiety"
    elif has_any(text, ["没感觉", "木木的", "麻木", "浑浑噩噩", "提不起劲", "发呆", "分不清", "什么感觉都"]):
        emotion = "hopelessness"
    elif has_any(text, ["好烦", "特别烦", "烦死", "烦透", "火大", "气得", "憋屈", "受不了"]) and not has_any(
        text, ["不麻烦", "没生气", "不像是生气", "懒得烦"]
    ):
        emotion = "anger"
    elif has_any(text, ["哭", "眼泪", "空落落", "难过", "走了", "去世"]):
        emotion = "sadness"
    elif has_any(text, ["说不上", "应该开心", "不知道自己"]):
        emotion = "neutral"
    else:
        emotion = "sadness"

    if intent == "crisis":
        intensity = 0.86
        if has_any(text, ["不想活", "消失", "没我大家都会轻松", "不用醒", "醒不过来"]):
            intensity = 0.92
        elif has_any(text, ["没什么意思", "没意义", "没劲", "就这样吧"]):
            intensity = 0.88
        if has_any(text, ["睡不着", "失眠", "没胃口", "吃不下", "头疼", "喘不上气", "胸口闷"]):
            intensity = max(intensity, 0.9)
    else:
        intensity = {
            "anxiety": 0.66,
            "sadness": 0.62,
            "anger": 0.58,
            "fear": 0.64,
            "shame": 0.60,
            "hopelessness": 0.72,
            "neutral": 0.42,
        }[emotion]
        if has_any(text, ["天天", "每天", "一直", "完全", "根本", "一口都", "整天"]) and has_any(
            text, ["睡不着", "失眠", "没胃口", "吃不下", "提不起劲", "发呆", "头疼"]
        ):
            intensity += 0.04
        if has_any(text, ["其实没什么大不了", "也没什么大不了", "没事", "不麻烦"]):
            intensity -= 0.05
        intensity = max(0.25, min(0.84, round(intensity, 4)))

    secondary_emotion = {
        "hopelessness": "sadness" if has_any(text, ["哭", "眼泪", "走了", "空落落", "分手", "离开"]) else "anxiety",
        "anxiety": "sadness" if has_any(text, ["分手", "走了", "空落落", "哭", "眼泪"]) else "fear",
        "sadness": "shame" if has_any(text, ["矫情", "没用", "不该"]) else "hopelessness",
        "shame": "sadness" if has_any(text, ["走了", "空落落", "哭"]) else "anxiety",
        "fear": "anxiety",
        "anger": "sadness",
        "neutral": "anxiety",
    }[emotion]
    secondary_intent = {
        "crisis": "venting",
        "seeking_advice": "venting",
        "seeking_empathy": "venting",
        "information_seeking": "seeking_advice",
        "self_disclosure": "venting",
        "venting": "self_disclosure",
    }[intent]

    return {
        "emotion_label": emotion,
        "emotion_scores": make_scores(EMOTIONS, emotion, secondary_emotion),
        "intent_label": intent,
        "intent_scores": make_scores(INTENTS, intent, secondary_intent),
        "intensity_score": round(intensity, 4),
        "risk_flag": intent == "crisis",
    }


def main() -> None:
    input_path = Path("data/processed/test_candidates_raw.md")
    output_path = Path("data/processed/mental_dialogue_test.jsonl")
    records = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw.update(classify(raw["text"]))
        if raw["id"] in OVERRIDES:
            override = OVERRIDES[raw["id"]]
            raw["emotion_label"] = str(override["emotion_label"])
            raw["intent_label"] = str(override["intent_label"])
            raw["intensity_score"] = float(override["intensity_score"])
            raw["risk_flag"] = bool(override["risk_flag"])
            secondary_emotion = "sadness" if raw["emotion_label"] == "hopelessness" else "anxiety"
            secondary_intent = "venting" if raw["intent_label"] != "venting" else "self_disclosure"
            raw["emotion_scores"] = make_scores(EMOTIONS, raw["emotion_label"], secondary_emotion)
            raw["intent_scores"] = make_scores(INTENTS, raw["intent_label"], secondary_intent)
        raw["source"] = "test_candidates_manual_annotation"
        records.append(raw)

    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    emotion_counts = Counter(record["emotion_label"] for record in records)
    intent_counts = Counter(record["intent_label"] for record in records)
    risk_true = sum(1 for record in records if record["risk_flag"])
    print(f"wrote {len(records)} records to {output_path}")
    print("emotion counts", dict(emotion_counts))
    print("intent counts", dict(intent_counts))
    print("risk_flag true", risk_true)


if __name__ == "__main__":
    main()
