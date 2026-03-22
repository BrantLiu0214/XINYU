# NLP Training Data Annotation Specification

This document defines the schema, label taxonomy, and annotation guidelines for the XinYu mental-health dialogue dataset. All training, validation, and test splits must conform to this specification. The NLP multitask model (`nlp/train/train_multitask.py`) reads data in this format directly.

## File Format

Each split is a UTF-8 encoded JSONL file (one JSON object per line). Blank lines are ignored.

```
data/
  raw/          ← original collected dialogues, unmodified
  processed/
    mental_dialogue_train.jsonl
    mental_dialogue_dev.jsonl
    mental_dialogue_test.jsonl
```

## Record Schema

```json
{
  "id": "string, unique sample identifier",
  "text": "string, raw user utterance in Simplified Chinese",
  "emotion_label": "string, one value from the emotion taxonomy",
  "emotion_scores": {
    "<label>": 0.0
  },
  "intent_label": "string, one value from the intent taxonomy",
  "intent_scores": {
    "<label>": 0.0
  },
  "intensity_score": 0.0,
  "risk_flag": false,
  "annotator_id": "string, optional",
  "source": "string, optional, origin of the sample"
}
```

All fields except `annotator_id` and `source` are required.

## Emotion Label Taxonomy

| Label | Chinese | Description | Example |
|-------|---------|-------------|---------|
| `anxiety` | 焦虑 | Worry, nervousness, anticipatory dread | "我一直担心考试，睡不着觉" |
| `sadness` | 悲伤 | Grief, loss, disappointment, low mood | "分手之后什么都提不起兴趣" |
| `anger` | 愤怒 | Frustration, irritation, rage, resentment | "他根本不理解我，我真的很烦" |
| `fear` | 恐惧 | Dread, panic, phobia, threat perception | "一想到未来我就很害怕" |
| `shame` | 羞耻 | Guilt, embarrassment, self-blame, inadequacy | "我觉得都是我的错，我太差劲了" |
| `hopelessness` | 绝望 | Despair, futility, giving up, suicidal affect | "我感觉怎么努力都没有意义了" |
| `neutral` | 中性 | Calm, factual, or mildly positive; no strong negative affect | "我今天想聊一下我的工作情况" |

**Rules:**
- Assign the single most prominent emotion. When two emotions co-occur at similar intensity, pick the one more likely to require clinical attention (e.g. prefer `hopelessness` over `sadness`).
- `emotion_scores` must sum to 1.0 across all labels. The primary label must have the highest score.
- Minimum primary label score: 0.40.

## Intent Label Taxonomy

| Label | Chinese | Description |
|-------|---------|-------------|
| `venting` | 倾诉 | Expressing emotions or experiences with no explicit request for guidance |
| `seeking_advice` | 寻求建议 | Asking for practical guidance, solutions, or next steps |
| `seeking_empathy` | 寻求共情 | Wanting to feel heard and understood; not asking for solutions |
| `crisis` | 危机表达 | Expressing suicidal ideation, self-harm intent, or immediate danger |
| `self_disclosure` | 自我披露 | Sharing personal background, history, or context |
| `information_seeking` | 寻求信息 | Asking factual or procedural questions about mental health topics |

**Rules:**
- Assign the single best-matching intent.
- `intent_scores` must sum to 1.0. The primary label must have the highest score.
- `crisis` must be assigned whenever any suicidal ideation, self-harm plan, or explicit hopeless-with-action language is present, regardless of other intent signals.

## Intensity Score

A continuous value in `[0.0, 1.0]` representing the overall emotional intensity of the utterance.

| Range | Interpretation |
|-------|---------------|
| 0.0 – 0.2 | Minimal: calm, matter-of-fact, almost no emotional charge |
| 0.2 – 0.5 | Mild: noticeable emotion but functional, not distressing |
| 0.5 – 0.7 | Moderate: clearly distressed, affecting daily function |
| 0.7 – 0.9 | High: acute distress, strong emotional language |
| 0.9 – 1.0 | Extreme: crisis language, overwhelming affect |

**Rule:** `intensity_score ≥ 0.85` should only be used when `emotion_label` is `hopelessness` or when `intent_label` is `crisis`.

## Risk Flag

A boolean that marks whether this utterance requires crisis-level intervention. Must be `true` when **any** of the following are present:

- Explicit suicidal ideation (e.g. "我不想活了", "结束一切")
- Self-harm intent or plan (e.g. "我想伤害自己")
- Request for means or methods
- Hopeless language combined with action framing (e.g. "没有意义了，我已经决定了")

`risk_flag: true` must always co-occur with `intent_label: "crisis"` and `intensity_score ≥ 0.80`.

## Annotation Guidelines

1. **Annotate the user turn only.** Assistant responses are not labelled.
2. **Use the full utterance for context.** Do not annotate individual sentences in isolation.
3. **When in doubt about risk, flag it.** False negatives (missing a crisis) are more harmful than false positives.
4. **Do not infer intent from prior turns.** Each utterance is labelled based on its own content.
5. **Resolve disagreements by majority vote** across at least 3 annotators. Samples with no majority agreement are excluded from training and flagged for review.

## Quality Criteria

- Inter-annotator agreement (Cohen's κ) ≥ 0.70 on `emotion_label` and `intent_label`.
- `risk_flag` recall among annotators ≥ 0.95 (prioritise recall over precision for safety).
- No sample with `risk_flag: true` may have `intensity_score < 0.75`.
- Distribution targets for `mental_dialogue_train.jsonl`:
  - No single emotion label exceeds 35% of samples.
  - `crisis` intent: at least 8% of samples (required for model recall on the safety-critical class).
  - `risk_flag: true`: at least 10% of samples.

## Minimum Dataset Size

| Split | Minimum samples |
|-------|----------------|
| train | 3 000 |
| dev | 500 |
| test | 500 |

The test split must not overlap with train or dev. Test samples must be held out before any annotation iteration to prevent label leakage.
