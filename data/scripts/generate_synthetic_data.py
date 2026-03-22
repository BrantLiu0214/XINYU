"""Domain corpus generation via Doubao (Volcengine Ark) API.

Generates train and dev splits with full annotation labels, plus a raw
(label-free) candidate pool for the test split that must be human-annotated
before use.

Usage:
    python data/scripts/generate_synthetic_data.py \\
        --output-dir data/processed \\
        --train-count 3000 \\
        --dev-count 500 \\
        --test-count 500 \\
        --api-key <key> \\
        --model <endpoint_id>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock


def _load_dotenv() -> None:
    """Load backend/.env into os.environ without overwriting existing values."""
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
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

import openai

# ── Label taxonomy ────────────────────────────────────────────────────────────

EMOTION_LABELS = ["anxiety", "sadness", "anger", "fear", "shame", "hopelessness", "neutral"]
INTENT_LABELS = [
    "venting", "seeking_advice", "seeking_empathy",
    "crisis", "self_disclosure", "information_seeking",
]

# Valid (emotion, intent) pairs.  Some combinations are clinically implausible
# (e.g. neutral × crisis) and are excluded to keep the corpus coherent.
_EXCLUDED: set[tuple[str, str]] = {
    ("neutral", "crisis"),
    ("neutral", "seeking_empathy"),
    ("hopelessness", "information_seeking"),
    ("hopelessness", "seeking_advice"),
}

VALID_PAIRS: list[tuple[str, str]] = [
    (e, i)
    for e in EMOTION_LABELS
    for i in INTENT_LABELS
    if (e, i) not in _EXCLUDED
]

# Distribution targets (fraction of total train samples).
# crisis intent ≥ 8%, risk_flag true ≥ 10%.
_PAIR_WEIGHTS: dict[tuple[str, str], float] = {
    # High-weight clinically important pairs
    ("hopelessness", "crisis"):     6.0,
    ("sadness", "crisis"):          2.5,
    ("fear", "crisis"):             1.5,
    ("hopelessness", "venting"):    4.0,
    ("hopelessness", "seeking_empathy"): 3.0,
    ("sadness", "venting"):         3.0,
    ("anxiety", "venting"):         3.0,
    ("anxiety", "seeking_advice"):  2.5,
    ("sadness", "seeking_empathy"): 2.5,
    ("anger", "venting"):           2.0,
    ("shame", "self_disclosure"):   2.0,
}

# Default weight for pairs not in the table above.
_DEFAULT_WEIGHT = 1.0


def _get_weight(pair: tuple[str, str]) -> float:
    return _PAIR_WEIGHTS.get(pair, _DEFAULT_WEIGHT)


# ── Intensity / risk heuristics ───────────────────────────────────────────────

def _default_intensity(emotion: str, intent: str) -> tuple[float, float]:
    """Return (intensity_mean, intensity_std) for this combination."""
    if intent == "crisis":
        return 0.88, 0.06
    if emotion == "hopelessness":
        return 0.75, 0.10
    if emotion in ("sadness", "fear", "shame"):
        return 0.60, 0.12
    if emotion in ("anxiety", "anger"):
        return 0.55, 0.12
    return 0.30, 0.10  # neutral / mild


def _default_risk_flag(emotion: str, intent: str, intensity: float) -> bool:
    if intent == "crisis":
        return True
    if emotion == "hopelessness" and intensity >= 0.80:
        return True
    return False


# ── Doubao API helpers ────────────────────────────────────────────────────────

_SYSTEM_PROMPT_LABELED = """\
你是一个心理对话数据标注专家。请根据给定的情绪类别和意图类别，生成一段真实、自然的用户发言，
模拟真实用户在心理支持聊天中可能说的话（简体中文，10~80字）。
同时，请为该发言生成完整的标注信息，以JSON格式返回，不要包含任何额外说明。

输出格式：
{
  "text": "用户发言内容",
  "emotion_scores": {"anxiety": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "shame": 0.0, "hopelessness": 0.0, "neutral": 0.0},
  "intent_scores": {"venting": 0.0, "seeking_advice": 0.0, "seeking_empathy": 0.0, "crisis": 0.0, "self_disclosure": 0.0, "information_seeking": 0.0},
  "intensity_score": 0.0,
  "risk_flag": false
}

注意事项：
- emotion_scores 各值之和必须为 1.0，最高分对应给定的情绪类别。
- intent_scores 各值之和必须为 1.0，最高分对应给定的意图类别。
- intensity_score 范围 0.0~1.0。
- 危机类别(intent=crisis)必须设 risk_flag=true，intensity_score >= 0.80。
- 只返回 JSON，不要有任何前缀或说明。
"""

_SYSTEM_PROMPT_RAW = """\
你是一个心理对话数据收集专家。请生成一段真实、自然的用户发言，
模拟真实用户在心理支持聊天中可能说的话（简体中文，10~80字）。
要求内容有一定挑战性或模糊性，适合作为标注测试样本。
只返回发言文本本身，不要有任何前缀、说明或引号。
"""


def _call_api(
    client: openai.OpenAI,
    model: str,
    system: str,
    user: str,
    max_retries: int = 3,
) -> str | None:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.9,
                max_tokens=300,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            wait = 2 * (attempt + 1)
            print(f"  API error (attempt {attempt + 1}/{max_retries}): {exc}. Retrying in {wait}s…")
            time.sleep(wait)
    return None


def _parse_labeled(raw: str, emotion: str, intent: str) -> dict | None:
    """Parse and validate a labeled generation response."""
    import re
    try:
        raw = raw.strip()
        # Strip <think>...</think> blocks emitted by reasoning models
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Strip markdown code fences if present
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
    if not text or len(text) < 5:
        return None

    # Validate and normalise emotion_scores
    e_scores = {k: float(obj["emotion_scores"].get(k, 0.0)) for k in EMOTION_LABELS}
    e_sum = sum(e_scores.values())
    if e_sum <= 0:
        return None
    e_scores = {k: round(v / e_sum, 4) for k, v in e_scores.items()}
    # Ensure the requested emotion is the highest
    if max(e_scores, key=lambda k: e_scores[k]) != emotion:
        e_scores[emotion] = max(e_scores.values()) + 0.05
        e_sum2 = sum(e_scores.values())
        e_scores = {k: round(v / e_sum2, 4) for k, v in e_scores.items()}

    # Validate and normalise intent_scores
    i_scores = {k: float(obj["intent_scores"].get(k, 0.0)) for k in INTENT_LABELS}
    i_sum = sum(i_scores.values())
    if i_sum <= 0:
        return None
    i_scores = {k: round(v / i_sum, 4) for k, v in i_scores.items()}
    if max(i_scores, key=lambda k: i_scores[k]) != intent:
        i_scores[intent] = max(i_scores.values()) + 0.05
        i_sum2 = sum(i_scores.values())
        i_scores = {k: round(v / i_sum2, 4) for k, v in i_scores.items()}

    intensity = max(0.0, min(1.0, float(obj["intensity_score"])))
    risk_flag = bool(obj["risk_flag"])

    # Enforce crisis invariant
    if intent == "crisis":
        risk_flag = True
        intensity = max(intensity, 0.80)

    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "emotion_label": emotion,
        "emotion_scores": e_scores,
        "intent_label": intent,
        "intent_scores": i_scores,
        "intensity_score": round(intensity, 4),
        "risk_flag": risk_flag,
        "source": "synthetic_doubao",
    }


def _compute_target_counts(
    pairs: list[tuple[str, str]], total: int
) -> dict[tuple[str, str], int]:
    weights = [_get_weight(p) for p in pairs]
    w_sum = sum(weights)
    raw = [w / w_sum * total for w in weights]
    counts = [max(1, int(r)) for r in raw]
    diff = total - sum(counts)
    # Distribute remainder to the pairs with the largest fractional parts
    fracs = sorted(range(len(raw)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    for i in range(abs(diff)):
        counts[fracs[i % len(fracs)]] += 1 if diff > 0 else -1
    return {pair: cnt for pair, cnt in zip(pairs, counts)}


def _check_distribution(records: list[dict], split_name: str) -> bool:
    total = len(records)
    if total == 0:
        return False

    emotion_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    risk_true = 0

    for r in records:
        emotion_counts[r["emotion_label"]] = emotion_counts.get(r["emotion_label"], 0) + 1
        intent_counts[r["intent_label"]] = intent_counts.get(r["intent_label"], 0) + 1
        if r.get("risk_flag"):
            risk_true += 1

    print(f"\n[{split_name}] {total} samples — distribution:")
    print("  Emotions:")
    ok = True
    for label in EMOTION_LABELS:
        pct = emotion_counts.get(label, 0) / total
        flag = " ← EXCEEDS 35%" if pct > 0.35 else ""
        print(f"    {label:<16}: {pct:.1%}{flag}")
        if pct > 0.35:
            ok = False

    print("  Intents:")
    crisis_pct = intent_counts.get("crisis", 0) / total
    flag = " ← BELOW 8%" if crisis_pct < 0.08 else ""
    for label in INTENT_LABELS:
        pct = intent_counts.get(label, 0) / total
        print(f"    {label:<22}: {pct:.1%}")
    if crisis_pct < 0.08:
        print(f"  WARNING: crisis intent {crisis_pct:.1%} < 8%{flag}")
        ok = False

    risk_pct = risk_true / total
    flag = " ← BELOW 10%" if risk_pct < 0.10 else ""
    print(f"  risk_flag=True: {risk_pct:.1%}{flag}")
    if risk_pct < 0.10:
        ok = False

    return ok


def generate_labeled_split(
    client: openai.OpenAI,
    model: str,
    target_counts: dict[tuple[str, str], int],
    split_name: str,
    output_path: Path,
    force: bool,
    workers: int = 20,
) -> list[dict]:
    if output_path.exists() and not force:
        print(f"  {output_path} already exists; skipping (use --force to regenerate).", flush=True)
        records = []
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    # Build the full list of (emotion, intent) tasks
    tasks: list[tuple[str, str]] = []
    for (emotion, intent), count in target_counts.items():
        tasks.extend([(emotion, intent)] * count)

    total_target = len(tasks)
    records: list[dict] = []
    lock = Lock()
    completed = [0]  # mutable counter shared across threads

    _PERSONAS = [
        "在校大学生", "职场新人", "中年上班族", "家庭主妇", "高中生", "自由职业者",
        "退休老人", "研究生", "创业者", "医护人员", "教师", "外来务工者",
        "单亲父母", "留学生", "应届毕业生", "离异中年人", "农村进城青年", "全职妈妈",
        "体制内员工", "艺术从业者", "运动员", "残障人士", "丧亲者", "慢性病患者",
    ]
    _SCENARIOS = [
        "感情问题", "工作压力", "家庭矛盾", "学业压力", "经济困难", "健康问题",
        "人际关系", "自我认同", "生活迷茫", "失眠困扰", "孤独感", "未来规划",
        "亲子冲突", "职场霸凌", "分手或失恋", "考试失败", "失业或裁员", "重大疾病",
        "丧亲之痛", "移居适应", "创伤后应激", "被孤立排斥", "身份认同危机", "长期照护压力",
    ]

    def _generate_one(emotion: str, intent: str) -> dict | None:
        persona = random.choice(_PERSONAS)
        scenario = random.choice(_SCENARIOS)
        user_prompt = (
            f"情绪类别：{emotion}\n意图类别：{intent}\n"
            f"角色背景：{persona}，正在经历{scenario}\n"
            f"请生成对应的用户发言及标注。"
        )
        raw = _call_api(client, model, _SYSTEM_PROMPT_LABELED, user_prompt)
        if raw is None:
            return None
        rec = _parse_labeled(raw, emotion, intent)
        if rec is None:
            raw2 = _call_api(client, model, _SYSTEM_PROMPT_LABELED, user_prompt)
            if raw2:
                rec = _parse_labeled(raw2, emotion, intent)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_generate_one, e, i): (e, i) for e, i in tasks}
        for future in as_completed(futures):
            rec = future.result()
            with lock:
                completed[0] += 1
                if rec:
                    records.append(rec)
                if completed[0] % 100 == 0:
                    print(f"  [{split_name}] {completed[0]}/{total_target} done, "
                          f"{len(records)} valid…", flush=True)

    # Deduplicate by exact text match before writing
    seen: set[str] = set()
    unique_records: list[dict] = []
    for rec in records:
        if rec["text"] not in seen:
            seen.add(rec["text"])
            unique_records.append(rec)
    n_dupes = len(records) - len(unique_records)
    if n_dupes:
        pct = n_dupes / len(records) * 100
        print(f"  [{split_name}] Removed {n_dupes} exact duplicates ({pct:.1f}%).", flush=True)
        if pct > 5:
            print(f"  WARNING: >5% duplicates removed — consider regenerating with --force.",
                  flush=True)
    records = unique_records

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    ok = _check_distribution(records, split_name)
    if ok:
        print(f"  All distribution constraints satisfied.", flush=True)
    else:
        print(f"  WARNING: one or more distribution constraints violated.", flush=True)
    return records


def generate_raw_test_candidates(
    client: openai.OpenAI,
    model: str,
    count: int,
    output_path: Path,
    force: bool,
    workers: int = 50,
) -> None:
    """Generate raw utterance texts only — no labels — for human annotation."""
    if output_path.exists() and not force:
        print(f"  {output_path} already exists; skipping (use --force to regenerate).")
        return

    texts: list[dict] = []
    lock = Lock()
    completed = [0]
    print(f"\n[test-raw] Generating {count} candidate texts (label-free)…", flush=True)

    # 9 personas × 9 scenarios = 81 base combos — ~6 samples/combo for 500 calls,
    # consistent with Set A's density (24×24=576 for 3500 samples).
    # 8 difficulty prompts overlay independently; they do not multiply the combo count.
    _difficulty_prompts = [
        "情绪模糊或自相矛盾，读者难以判断其主要情绪。",
        "表达负面情绪但意图不明确，不清楚是在倾诉、求助还是寻求共鸣。",
        "可能需要关注但未明确表达危机，存在隐性风险信号。",
        "用日常琐事掩盖真实情绪困扰，表面平淡但暗含压力。",
        "刻意轻描淡写或自我否定真实痛苦，如'其实没什么大不了'。",
        "被动表达危机意念但未直接求助，如'活着也没什么意思'。",
        "用躯体化症状（失眠、头痛、食欲不振）描述心理困扰。",
        "经历长期慢性压力已经麻木或情感钝化，语气平静但内容沉重。",
    ]

    _PERSONAS_RAW = [
        "在校大学生", "职场新人", "中年上班族", "高中生", "研究生",
        "单亲父母", "留学生", "医护人员", "慢性病患者",
    ]
    _SCENARIOS_RAW = [
        "工作压力", "学业压力", "家庭矛盾", "感情问题", "经济困难",
        "人际关系", "自我认同", "生活迷茫", "丧亲之痛",
    ]

    def _gen_one_raw(_: int) -> str | None:
        persona = random.choice(_PERSONAS_RAW)
        scenario = random.choice(_SCENARIOS_RAW)
        difficulty = random.choice(_difficulty_prompts)
        user_prompt = (
            f"角色背景：{persona}，正在经历{scenario}\n"
            f"请生成一段符合以下要求的用户发言：{difficulty}"
        )
        return _call_api(client, model, _SYSTEM_PROMPT_RAW, user_prompt)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_gen_one_raw, i): i for i in range(count)}
        for future in as_completed(futures):
            raw = future.result()
            with lock:
                completed[0] += 1
                if raw:
                    texts.append({"id": str(uuid.uuid4()), "text": raw.strip()})
                if completed[0] % 50 == 0:
                    print(f"  [test-raw] {completed[0]}/{count} texts done…", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in texts:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(
        f"[test-raw] {len(texts)} candidate texts written to {output_path}\n"
        f"  NOTE: These texts have NO labels. A human annotator must label all\n"
        f"  records following data/annotation-spec.md before creating\n"
        f"  data/processed/mental_dialogue_test.jsonl."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate domain training corpus via Doubao API")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--train-count", type=int, default=3000, dest="train_count")
    parser.add_argument("--dev-count", type=int, default=500, dest="dev_count")
    parser.add_argument("--test-count", type=int, default=500, dest="test_count",
                        help="Number of raw test candidate texts (no labels)")
    parser.add_argument("--api-key", default=os.getenv("XINYU_DOUBAO_API_KEY", ""),
                        dest="api_key")
    parser.add_argument("--model", default=os.getenv("XINYU_DOUBAO_MODEL", ""),
                        dest="model")
    parser.add_argument("--base-url",
                        default=os.getenv("XINYU_DOUBAO_BASE_URL", "https://ark.volcengine.com/api/v3"),
                        dest="base_url")
    parser.add_argument("--workers", type=int, default=50,
                        help="Number of concurrent API threads (default: 50)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    if not args.api_key or not args.model:
        print("ERROR: --api-key and --model are required (or set XINYU_DOUBAO_API_KEY / "
              "XINYU_DOUBAO_MODEL env vars).")
        raise SystemExit(1)

    client = openai.OpenAI(api_key=args.api_key, base_url=args.base_url)
    out_dir = Path(args.output_dir)
    raw_dir = Path("data/raw")

    train_counts = _compute_target_counts(VALID_PAIRS, args.train_count)
    dev_counts = _compute_target_counts(VALID_PAIRS, args.dev_count)

    print(f"Generating train split ({args.train_count} samples)…", flush=True)
    generate_labeled_split(
        client, args.model, train_counts, "train",
        out_dir / "mental_dialogue_train.jsonl", args.force,
        workers=args.workers,
    )

    print(f"\nGenerating dev split ({args.dev_count} samples)…", flush=True)
    generate_labeled_split(
        client, args.model, dev_counts, "dev",
        out_dir / "mental_dialogue_dev.jsonl", args.force,
        workers=args.workers,
    )

    generate_raw_test_candidates(
        client, args.model, args.test_count,
        raw_dir / "test_candidates_raw.jsonl", args.force,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
