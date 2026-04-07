"""Shared crisis keyword sets used by both RiskService and RealNLPService.

Keeping these in one place ensures that keyword-hit detection in the NLP
layer and risk evaluation in the risk layer always scan for exactly the same
terms.  Neither module should define its own copy of these sets.
"""
from __future__ import annotations

# Presence of any L3 keyword in an utterance directly triggers the crisis path.
L3_KEYWORDS: frozenset[str] = frozenset({
    "自杀", "自伤", "不想活", "结束生命", "结束一切",
    "伤害自己", "死了算了", "去死", "想死", "轻生",
})

# Presence of L2 keywords contributes to moderate-risk escalation.
L2_KEYWORDS: frozenset[str] = frozenset({
    "崩溃", "绝望", "撑不下去", "活不下去",
    "消失", "放弃一切", "没有意义", "活着没意思", "活着没什么意思",
})
