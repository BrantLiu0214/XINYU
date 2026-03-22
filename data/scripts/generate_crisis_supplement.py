"""Generate supplementary crisis training examples covering 7 implicit crisis patterns
that the model fails to detect in the tier-2 crisis probe.

Run from project root:
    python data/scripts/generate_crisis_supplement.py
"""
from __future__ import annotations

import json
import os
import random
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock


def _load_dotenv() -> None:
    env_path = Path(__file__).parent.parent.parent / "backend" / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


_load_dotenv()
import openai

# ── 7 implicit crisis patterns ────────────────────────────────────────────────

_PATTERNS = [
    {
        "name": "passive_death_wish",
        "desc": "被动死亡意愿：不直接说想死，而是表达如果能睡过去不再醒来就好了、或者消极地等待死亡",
        "examples": ["要是就这样睡过去再也不用醒就好了", "有时候很羡慕那些走了的人"],
    },
    {
        "name": "location_method_ideation",
        "desc": "地点/方法幻想：站在高处或危险地点，脑子里浮现出跳下去或以某种方式结束的念头",
        "examples": ["站在阳台边上发呆，脑子里一直在想跳下去是不是就轻松了"],
    },
    {
        "name": "access_to_means",
        "desc": "手段可及：描述手边有可用于自伤或自杀的工具（药物、刀、绳子等），并凝视或思考使用它",
        "examples": ["手边就放着之前剩下的药，刚刚盯着它们看了很久"],
    },
    {
        "name": "burdensomeness",
        "desc": "自我是负担：认为自己活着只是在拖累家人或所有人，如果消失了他们会更好",
        "examples": ["我觉得自己活着只是在拖累所有人，要是没有我他们应该都会轻松很多"],
    },
    {
        "name": "wanting_to_disappear",
        "desc": "想消失/消除自己：不直接说死，而是说想消失、不想再继续存在、反正也没有人在意",
        "examples": ["已经不止一次想过消失掉算了，反正也不会有人真正在意"],
    },
    {
        "name": "self_harm_act",
        "desc": "自我伤害行为：已经实施了某种程度的自我伤害动作，或刚刚阻止了自己但仍在挣扎",
        "examples": ["刚刚把刀拿出来又放回去，现在整个人都在抖，不知道还能不能控制住自己"],
    },
    {
        "name": "concrete_plan_timing",
        "desc": "具体计划+时间点：已经决定好了行动，并给出了具体的时间或条件（等大家睡了、等今晚）",
        "examples": ["已经想好了，等今晚所有人睡着以后我就不想再继续了"],
    },
]

_PERSONAS = [
    "在校大学生", "职场新人", "中年上班族", "高中生", "研究生",
    "单亲父母", "留学生", "医护人员", "慢性病患者", "离异中年人",
    "退休老人", "丧亲者", "应届毕业生", "自由职业者", "全职妈妈",
]

_SYSTEM_PROMPT = """\
你是一个心理危机对话数据标注专家。请根据给定的危机表达模式，生成一段真实、自然的用户发言，
模拟真实用户在心理支持聊天中可能说的话（简体中文，15~80字）。
发言必须体现给定的危机模式，但语言要自然，不要刻意用"我想死"等过于直白的表达。
以JSON格式返回，不要包含任何额外说明。

输出格式：
{
  "text": "用户发言内容",
  "emotion_scores": {"anxiety": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "shame": 0.0, "hopelessness": 0.0, "neutral": 0.0},
  "intent_scores": {"venting": 0.0, "seeking_advice": 0.0, "seeking_empathy": 0.0, "crisis": 0.0, "self_disclosure": 0.0, "information_seeking": 0.0},
  "intensity_score": 0.0,
  "risk_flag": true
}

规则：
- emotion_scores 各值之和必须为1.0，主情绪为 hopelessness 或 sadness。
- intent_scores 各值之和必须为1.0，crisis 必须是最高分。
- intensity_score ≥ 0.85。
- risk_flag 必须为 true。
- 只返回 JSON，不要有任何前缀或说明。
"""

EMOTION_LABELS = ["anxiety", "sadness", "anger", "fear", "shame", "hopelessness", "neutral"]
INTENT_LABELS = ["venting", "seeking_advice", "seeking_empathy", "crisis", "self_disclosure", "information_seeking"]


def _parse(raw: str) -> dict | None:
    try:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None

    required = {"text", "emotion_scores", "intent_scores", "intensity_score", "risk_flag"}
    if not required.issubset(obj):
        return None
    text = str(obj["text"]).strip()
    if len(text) < 10:
        return None

    e_scores = {k: float(obj["emotion_scores"].get(k, 0.0)) for k in EMOTION_LABELS}
    e_sum = sum(e_scores.values())
    if e_sum <= 0:
        return None
    e_scores = {k: round(v / e_sum, 4) for k, v in e_scores.items()}

    i_scores = {k: float(obj["intent_scores"].get(k, 0.0)) for k in INTENT_LABELS}
    i_sum = sum(i_scores.values())
    if i_sum <= 0:
        return None
    i_scores = {k: round(v / i_sum, 4) for k, v in i_scores.items()}
    # Enforce crisis as highest intent
    if max(i_scores, key=lambda k: i_scores[k]) != "crisis":
        i_scores["crisis"] = max(i_scores.values()) + 0.1
        i_sum2 = sum(i_scores.values())
        i_scores = {k: round(v / i_sum2, 4) for k, v in i_scores.items()}

    intensity = max(0.85, min(1.0, float(obj.get("intensity_score", 0.9))))

    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "emotion_label": max(e_scores, key=lambda k: e_scores[k]),
        "emotion_scores": e_scores,
        "intent_label": "crisis",
        "intent_scores": i_scores,
        "intensity_score": round(intensity, 4),
        "risk_flag": True,
        "source": "synthetic_crisis_supplement",
    }


def generate(n_per_pattern: int = 15, workers: int = 50) -> list[dict]:
    api_key = os.getenv("XINYU_DOUBAO_API_KEY", "")
    model = os.getenv("XINYU_DOUBAO_MODEL", "")
    base_url = os.getenv("XINYU_DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

    if not api_key or not model:
        raise SystemExit("ERROR: XINYU_DOUBAO_API_KEY and XINYU_DOUBAO_MODEL must be set.")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # Build task list: n_per_pattern × 7 patterns
    tasks: list[dict] = []
    for pattern in _PATTERNS:
        for _ in range(n_per_pattern):
            tasks.append(pattern)

    records: list[dict] = []
    lock = Lock()
    completed = [0]

    def _gen_one(pattern: dict) -> dict | None:
        persona = random.choice(_PERSONAS)
        example = random.choice(pattern["examples"])
        user_prompt = (
            f"危机模式：{pattern['desc']}\n"
            f"角色背景：{persona}\n"
            f"参考示例风格（不要直接复制）：{example}\n"
            f"请生成一段不同的、自然的用户发言及标注。"
        )
        for _ in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.95,
                    max_tokens=300,
                )
                raw = resp.choices[0].message.content.strip()
                rec = _parse(raw)
                if rec:
                    return rec
            except Exception as exc:
                print(f"  API error: {exc}", flush=True)
        return None

    print(f"Generating {len(tasks)} crisis supplement examples ({n_per_pattern} per pattern × 7 patterns)…", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_gen_one, t): t for t in tasks}
        for future in as_completed(futures):
            rec = future.result()
            with lock:
                completed[0] += 1
                if rec:
                    records.append(rec)
                if completed[0] % 20 == 0:
                    print(f"  {completed[0]}/{len(tasks)} done, {len(records)} valid…", flush=True)

    # Deduplicate
    seen: set[str] = set()
    unique = []
    for r in records:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique.append(r)
    print(f"Generated {len(unique)} unique records ({len(records) - len(unique)} duplicates removed).", flush=True)
    return unique


def main() -> None:
    records = generate(n_per_pattern=15, workers=50)

    # Append to train split
    train_path = Path("data/processed/mental_dialogue_train.jsonl")
    with open(train_path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Appended {len(records)} records to {train_path}.", flush=True)
    print(f"New train size: {sum(1 for _ in open(train_path, encoding='utf-8'))} records.", flush=True)


if __name__ == "__main__":
    main()
