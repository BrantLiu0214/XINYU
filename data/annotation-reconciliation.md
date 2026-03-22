# Annotation Reconciliation Notes

This note records the reconciled decision rules derived from the 100-sample IAA review on 2026-03-21.

## Scope

- Shared samples reviewed: 100
- Disagreement cases exported: 39
- Reconciled file: `data/processed/iaa_reconciled_100.jsonl`

## Reconciled Rules

1. `hopelessness` / `crisis` high-recall boundary

- Phrases such as `活着没意思`, `不用醒`, `消失`, and similar "life is not worth continuing" framing should be escalated toward `hopelessness`.
- If the utterance includes passive death wish, disappearance framing, or high-risk fixation such as `高处发呆` together with clear distress, prefer `intent_label: "crisis"` for recall.
- `睡过去再也醒不来也不错` is treated as `crisis`.

2. `venting` vs `self_disclosure`

- If the speaker is clearly expressing current distress, burden, or functional impact, prefer `venting`.
- Do not switch to `self_disclosure` only because the utterance includes background context.
- Reserve `self_disclosure` for descriptive sharing where the current turn is mainly giving history/context rather than actively expressing present pain.

3. `sadness` vs `hopelessness`

- For `麻木`, `没感觉`, `没盼头`, `提不起劲`, `空掉`, `耗着`, prioritize `hopelessness` when the overall tone implies futility or emotional shutdown.
- Use `sadness` when the dominant signal is grief, loss, emptiness, or low mood without clear futility / giving-up framing.

## Operational Outcome

- `mental_dialogue_test.jsonl` vs `iaa_reconciled_100.jsonl`
  - emotion kappa: `0.7440`
  - intent kappa: `0.7455`
- `iaa_annotator_b.jsonl` vs `iaa_reconciled_100.jsonl`
  - emotion kappa: `0.8388`
  - intent kappa: `0.6360`

The remaining intent gap is mainly a `venting` vs `self_disclosure` threshold issue on B-side annotations. Training should use the reconciled file or apply the same rules before expanding annotation.
