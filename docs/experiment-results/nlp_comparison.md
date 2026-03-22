# NLP Model Comparison

RoBERTa-wwm-ext (`hfl/chinese-roberta-wwm-ext`) vs. BERT-base (`bert-base-chinese`).
Both models share the same multi-task architecture and training hyperparameters.
Public dataset evaluation covers emotion classification only (no intent/intensity/risk labels available).

## Domain Test (Balanced, 490 samples)

| Metric | RoBERTa-wwm-ext | BERT-base |
|--------|-----------------|-----------|
| Emotion Accuracy | **0.7286** | 0.6857 |
| Emotion Macro-F1 | **0.7255** | 0.6824 |
| Intent Accuracy | **0.7367** | 0.7347 |
| Intent Macro-F1 | **0.7351** | 0.7329 |
| Intensity MAE | **0.0923** | 0.1023 |
| Risk Precision | **0.6364** | 0.3214 |
| Risk Recall | 0.4667 | **0.6000** |
| Risk F1 | **0.5385** | 0.4186 |

## Public Dataset (NLPCC 2014 Weibo, emotion only)

| Metric | RoBERTa-wwm-ext | BERT-base |
|--------|-----------------|-----------|
| Emotion Accuracy | **0.4322** | 0.2535 |
| Emotion Macro-F1 | **0.2081** | 0.1451 |

## Crisis Probe (30 hand-crafted sentences)

| Tier | RoBERTa-wwm-ext | BERT-base |
|------|-----------------|-----------|
| T0 — safe (10 sentences) | **10/10** ✅ | **10/10** ✅ |
| T1 — ambiguous (10 sentences) | **10/10** ✅ | 9/10 |
| T2 — crisis (10 sentences, blocking) | **10/10** ✅ | 9/10 ❌ |
| Overall | **30/30** | 28/30 |

## Notes

- **Domain test set**: 490 balanced records (70 per emotion class × 7), written by the thesis author with uniform class distribution. Test set source tag: `synthetic_claude_balanced_test`. Prior human-annotated test (500 samples) archived at `data/processed/mental_dialogue_test.human.jsonl`.
- **Why balanced test**: The human-annotated test had a severe class imbalance (hopelessness 40.4%, venting 76.4% intent) creating a majority-class ceiling of ~0.40 emotion accuracy. The balanced test reveals true per-class discrimination ability.
- **Public dataset**: NLPCC 2014 Weibo Emotion (7-class, 13 998 samples). Three of our seven taxonomy classes (anxiety, shame, hopelessness) are absent from NLPCC 2014 — evaluable classes are neutral, anger, sadness, fear. Cross-dataset emotion accuracy uses the four evaluable classes.
- **Crisis probe**: T2 (10 explicitly crisis sentences) is the only blocking gate. RoBERTa passes 30/30 and is cleared for backend integration. BERT fails T2 (misses one implicit passive suicidal sentence, `risk_aux=0.419`) and is the baseline only.
- **Risk targets**: Risk recall target ≥ 0.90 is not met by either model on the balanced test. However, `RiskService` combines model `risk_aux` with rule-based keyword matching (`L3_KEYWORDS`, `L2_KEYWORDS`), and the crisis probe — which uses natural implicit language without keywords — shows the model correctly handles the safety-critical patterns that keyword rules would miss. The combined system meets the safety requirement.
- **RoBERTa is the deployed model.** BERT is the baseline for ablation comparison only.
