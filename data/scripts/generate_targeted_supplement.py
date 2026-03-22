"""Append ~120 targeted training records to address the three confusion clusters
identified in the RoBERTa confusion matrix:

  1. shame → predicted as neutral  (58% miss rate)
     Fix: 50 shame records with quiet, factual-sounding self-blame tone
  2. self_disclosure → predicted as venting  (90% miss rate)
     Fix: 40 self_disclosure records with clearly descriptive, low-emotion tone
  3. anxiety → predicted as sadness  (42% miss rate)
     Fix: 30 anxiety records with clear worry/tension language

Run from project root:
    python data/scripts/generate_targeted_supplement.py
"""
from __future__ import annotations
import json
import uuid
from pathlib import Path

EMOTION_LABELS = ["anxiety", "sadness", "anger", "fear", "shame", "hopelessness", "neutral"]
INTENT_LABELS  = ["venting", "seeking_advice", "seeking_empathy", "crisis",
                  "self_disclosure", "information_seeking"]


def r(primary: str, labels: list[str], secondary: dict[str, float]) -> dict[str, float]:
    scores = {k: 0.0 for k in labels}
    scores.update(secondary)
    scores[primary] = round(1.0 - sum(secondary.values()), 4)
    total = sum(scores.values())
    if abs(total - 1.0) > 1e-9:
        scores[primary] = round(scores[primary] + (1.0 - total), 4)
    return scores


def rec(text, el, escores, il, iscores, intensity, risk=False):
    return {"id": str(uuid.uuid4()), "text": text,
            "emotion_label": el, "emotion_scores": escores,
            "intent_label": il, "intent_scores": iscores,
            "intensity_score": round(intensity, 4),
            "risk_flag": risk, "source": "synthetic_claude_targeted"}


RECORDS: list[dict] = []

# ══════════════════════════════════════════════════════════════════════════════
# 1. SHAME × various intents — quiet, factual-sounding self-blame (50 records)
#    Key: these texts look calm/neutral on surface but contain self-attribution
#    ("都是我的问题", "本来就是我太…", "我自己的毛病", "怪我自己", "我确实不行")
# ══════════════════════════════════════════════════════════════════════════════

# shame × venting (20) — expressing self-blame without asking for anything
SHAME_VENTING = [
    ("可能就是我自己不够努力，没什么好说的。", 0.45),
    ("也怪我自己当时没想清楚，现在说这些也没用了。", 0.48),
    ("本来就是我性格上有问题，很多事搞不好也正常。", 0.50),
    ("都是我自己的毛病，别人也没有义务迁就我。", 0.52),
    ("我确实不行，承认这一点反而心里能踏实一点。", 0.47),
    ("这种事情换别人早就处理好了，还是我太笨。", 0.53),
    ("我就是抗压能力差，遇到点事就撑不住，没什么可抱怨的。", 0.55),
    ("都是我自己拖延惯了，到现在还是老样子，改不了。", 0.49),
    ("本来就是我太玻璃心，这点小事都扛不住，挺没用的。", 0.54),
    ("我应该早就发现问题的，没发现是我粗心，没理由怪别人。", 0.50),
    ("这次失败很大程度上是我自己准备不足，认了。", 0.46),
    ("我总是把关系搞砸，可能真的是我的问题。", 0.55),
    ("别人都能适应，就我觉得困难，说明还是我的问题多。", 0.52),
    ("出了这种事首先是我自己的责任，我知道。", 0.48),
    ("我这个人太敏感，很容易让别人觉得难相处，是我的问题。", 0.53),
    ("不善言辞是我从小就有的毛病，现在也只能这样。", 0.44),
    ("说到底还是我自己没本事，处处需要别人帮忙，很拖累人。", 0.57),
    ("这段关系没走下去，我觉得主要还是我的问题更多。", 0.51),
    ("我脾气不好，自己清楚，总是让家里人受气，挺对不住他们的。", 0.56),
    ("也许是我要求太多了，对别人来说我可能真的很难相处。", 0.50),
]
for text, inten in SHAME_VENTING:
    RECORDS.append(rec(text, "shame",
                       r("shame", EMOTION_LABELS, {"sadness": 0.08, "neutral": 0.06}),
                       "venting",
                       r("venting", INTENT_LABELS, {"self_disclosure": 0.10}),
                       inten))

# shame × self_disclosure (15) — describing oneself with self-blame framing
SHAME_SD = [
    ("我从小就觉得自己比别人差一截，不是谦虚，就是客观感受。", 0.42),
    ("我是那种很难在人群里放松的人，总觉得自己哪里不对劲。", 0.44),
    ("我一直觉得自己的工作能力不算强，靠运气撑过来的多。", 0.43),
    ("我有个毛病，总是把事情搞砸之后第一个怪自己。", 0.48),
    ("我这人情商不高，很多时候说话会让别人不舒服，自己事后才反应过来。", 0.47),
    ("我从高中开始就觉得自己和周围人不太一样，但说不出哪里不一样。", 0.40),
    ("我平时给人的印象好像都是还好，但我自己知道自己能力有限。", 0.43),
    ("我和前任分手是因为我当时不够成熟，现在想想很多地方是我的问题。", 0.50),
    ("我是那种什么都想做好但什么都做得一般的人，有点痛苦。", 0.52),
    ("我意识到自己有很多地方需要改，但改起来总是不成功。", 0.49),
    ("我在家里排行老小，从小被保护，所以很多事情我确实没有别人能干。", 0.38),
    ("我有表达障碍，说不出口的话很多，所以常常被人误解，怪我自己。", 0.47),
    ("我这人做事容易犯错，被批评多了就习惯了，觉得理所当然。", 0.48),
    ("我一直觉得自己在这份工作里就是那种可有可无的人。", 0.45),
    ("我很难接受表扬，每次被夸我都觉得对方不了解我真实水平。", 0.46),
]
for text, inten in SHAME_SD:
    RECORDS.append(rec(text, "shame",
                       r("shame", EMOTION_LABELS, {"sadness": 0.06, "neutral": 0.08}),
                       "self_disclosure",
                       r("self_disclosure", INTENT_LABELS, {"venting": 0.08}),
                       inten))

# shame × seeking_empathy (15) — self-blame but wanting to be understood
SHAME_SE = [
    ("我老是觉得自己太差劲，有时候真的不知道能不能变好，有人懂这种感觉吗？", 0.58),
    ("我一直在怪自己，但其实我真的很努力了，只是结果不好，有人理解吗？", 0.56),
    ("我知道是我的问题，但说出来被人说'那你努力改啊'就很烦，只是想让人听一听。", 0.55),
    ("每次出了事我都第一个自我检讨，但检讨完还是很难受，有人能懂吗？", 0.57),
    ("我很清楚自己哪里不好，但清楚了也没变好，这种感觉很憋屈，想找人说说。", 0.59),
    ("总是觉得自己拖累了别人，这种想法挥不掉，不知道有没有人有类似的感受。", 0.60),
    ("我对自己的要求挺高但结果总让自己失望，已经很习惯自责了，想聊聊。", 0.57),
    ("我知道说这些没用，就是需要有人听我说一声我已经很努力了。", 0.54),
    ("别人看我好像挺正常的，但我自己知道我总是在内心批评自己，有点累。", 0.56),
    ("我有时候会觉得连难过的资格都没有，因为是自己造成的，想找人说说。", 0.62),
    ("我不需要建议，就是想有人告诉我这不完全是我的错，可能吗？", 0.58),
    ("我把朋友关系搞砸了，一直在自责，只是想有人听一听，不评判那种。", 0.57),
    ("我觉得我给家人造成了很多麻烦，这个念头经常有，想找人聊聊。", 0.59),
    ("有时候羡慕那些不怎么自责的人，我这种自我批评的习惯真的很累，有人懂吗？", 0.56),
    ("我一直觉得自己本来就不够好，被人嫌弃也是正常的，只是想找个地方说说。", 0.61),
]
for text, inten in SHAME_SE:
    RECORDS.append(rec(text, "shame",
                       r("shame", EMOTION_LABELS, {"sadness": 0.10, "hopelessness": 0.06}),
                       "seeking_empathy",
                       r("seeking_empathy", INTENT_LABELS, {"venting": 0.12}),
                       inten))

# ══════════════════════════════════════════════════════════════════════════════
# 2. SELF_DISCLOSURE × various emotions (40 records)
#    Key: clearly descriptive background, minimal emotional outpouring
#    Should NOT look like venting; has "我是…", "我一直…", "我的情况是…" framing
# ══════════════════════════════════════════════════════════════════════════════

# self_disclosure × anxiety (12)
SD_ANX = [
    ("我是去年刚换工作的，新环境还在适应，总有点担心做不好。", 0.32),
    ("我一直是个比较容易紧张的人，考试前后心跳总会加快。", 0.35),
    ("我是独居的，家人不在身边，有时候遇到事会比较慌，这是我的一个情况。", 0.33),
    ("我有点焦虑体质，这是我自己总结的，遇到不确定的事就会反复想。", 0.36),
    ("我的情况是这样的，刚从外地来，还没稳定下来，有一些不安全感。", 0.30),
    ("我在一家创业公司，岗位不太稳定，这件事让我一直有些担忧。", 0.34),
    ("我刚开始实习，不确定自己能不能留下来，这是我目前的状态。", 0.32),
    ("我是慢性病患者，需要长期用药，每次复诊前都会有些担心结果。", 0.38),
    ("我的情况是：我妈最近身体不太好，我在外地，一直有些放心不下。", 0.36),
    ("我目前在备考，压力不小，总担心复习进度跟不上，这是我的基本情况。", 0.35),
    ("我是一个比较依赖计划的人，计划被打乱就会有点焦虑，这是我的特点。", 0.30),
    ("我从事的是医护行业，值班频率高，不确定性多，这是我长期的一个背景压力。", 0.34),
]
for text, inten in SD_ANX:
    RECORDS.append(rec(text, "anxiety",
                       r("anxiety", EMOTION_LABELS, {"neutral": 0.12, "fear": 0.06}),
                       "self_disclosure",
                       r("self_disclosure", INTENT_LABELS, {"venting": 0.07}),
                       inten))

# self_disclosure × neutral (15)
SD_NEU = [
    ("我是大三学生，最近在找实习，这是我现在的主要状态。", 0.12),
    ("我在北京工作三年了，一个人租房，生活基本稳定，没什么特别的。", 0.10),
    ("我有两个孩子，老大上小学，老二还在幼儿园，每天就这样。", 0.11),
    ("我是做设计的，自由职业，收入不固定，生活比较随机。", 0.13),
    ("我父母在老家，我在外地，每周视频通话一次，这是我们的相处方式。", 0.12),
    ("我最近在学驾照，每周末去练车，这是我目前在做的事。", 0.10),
    ("我的工作是内容运营，主要对接几个品牌客户，工作量时多时少。", 0.11),
    ("我研究生刚毕业，进了国企，正式工作第一个月，一切都在摸索中。", 0.14),
    ("我是全职妈妈，孩子两岁，另一半在外地出差居多，我们这边情况就这样。", 0.13),
    ("我在一家中型公司做行政，工作相对规律，没什么太大起伏。", 0.10),
    ("我是理工科背景，现在在做数据分析，工作内容偏技术，不太接触人。", 0.11),
    ("我今年刚结婚，跟爱人在同一家公司，这是我现在的基本情况。", 0.10),
    ("我是家里的老大，下面有个弟弟，从小父母对我要求比较高。", 0.14),
    ("我现在在农村老家陪父母住，暂时没有工作，就是帮家里做做事。", 0.12),
    ("我大学读的中文系，现在在出版社工作，做编辑相关的，基本就这样。", 0.10),
]
for text, inten in SD_NEU:
    RECORDS.append(rec(text, "neutral",
                       r("neutral", EMOTION_LABELS, {"anxiety": 0.06, "sadness": 0.04}),
                       "self_disclosure",
                       r("self_disclosure", INTENT_LABELS, {"venting": 0.06}),
                       inten))

# self_disclosure × sadness (13)
SD_SAD = [
    ("我是丧亲者，父亲去年去世了，这是我目前状态的背景。", 0.45),
    ("我和爱人今年离婚了，现在一个人带孩子，这是我现在的生活。", 0.48),
    ("我大学好友上个月意外去世了，我还没有完全接受这件事。", 0.50),
    ("我被公司裁员了，已经失业三个月，一直在投简历，情况就这样。", 0.43),
    ("我跟了五年的男朋友去年底分手了，现在还在调整中，这是背景。", 0.47),
    ("我妈去年查出来有病，在治疗中，家里状态比以前紧张了很多。", 0.49),
    ("我有抑郁病史，两年前治疗过一次，现在基本稳定但没有完全好。", 0.46),
    ("我的孩子在特殊学校就读，这件事对我和爱人都有很大的影响。", 0.48),
    ("我去年经历了一场比较严重的车祸，身体恢复了，但心理上还有影响。", 0.47),
    ("我在外地打工很多年，今年第一次没有回家过年，心里有些不是滋味。", 0.42),
    ("我失去了一段很重要的友情，对方突然断联，我一直不明白原因。", 0.46),
    ("我奶奶前几个月走了，我们感情很好，这件事对我影响挺大的。", 0.49),
    ("我做了一个让自己很后悔的决定，时间已经过去了，但我还是会想起来。", 0.44),
]
for text, inten in SD_SAD:
    RECORDS.append(rec(text, "sadness",
                       r("sadness", EMOTION_LABELS, {"hopelessness": 0.10, "neutral": 0.06}),
                       "self_disclosure",
                       r("self_disclosure", INTENT_LABELS, {"venting": 0.08}),
                       inten))

# ══════════════════════════════════════════════════════════════════════════════
# 3. ANXIETY — clearly worried/tense language, distinct from sadness (30 records)
#    Key: forward-looking worry, physical tension signs, "担心/紧张/害怕/不安"
#    Must NOT use retrospective sadness language
# ══════════════════════════════════════════════════════════════════════════════

ANX_VENT = [
    ("体检报告有一项指标偏高，医生说观察，但我这几天一直在担心。", 0.58),
    ("下周要做一个很重要的汇报，一想到就心跳加速，感觉准备再多也不够。", 0.62),
    ("我妈最近老说胸闷，我一直在担心她，让她去医院她又不肯。", 0.60),
    ("最近总是晚上睡不着，脑子里转的都是还没解决的事情，停不下来。", 0.64),
    ("我在等一个很重要的结果，等待的过程比我预想的要难熬很多。", 0.59),
    ("我有个决定迟迟下不了，两个选项都有风险，反复想来想去很累。", 0.61),
    ("搬到新城市快一个月了，一直有点不踏实，总觉得还没站稳。", 0.57),
    ("我负责的项目进度落后了，每天早上睁眼第一件事就是担心这个。", 0.65),
    ("我已经连着好几周睡眠质量很差，不知道是压力还是别的原因，有点不安。", 0.63),
    ("论文答辩快到了，感觉心里一直绷着一根弦，没办法真正放松下来。", 0.66),
    ("朋友跟我说了一件事，我现在一直在为他担心，也不知道能帮上什么。", 0.58),
    ("我最近总是在想万一失业了怎么办，虽然现在还没有这个信号，就是控制不住。", 0.64),
    ("跟父母打电话总是心里堵，怕他们在老家出什么事，但也帮不上什么。", 0.60),
    ("我一直不确定自己选的这条路对不对，这种不确定感已经持续很久了。", 0.62),
    ("我身边一个朋友突然得了大病，这让我开始莫名担心自己的健康。", 0.61),
]
for text, inten in ANX_VENT:
    RECORDS.append(rec(text, "anxiety",
                       r("anxiety", EMOTION_LABELS, {"fear": 0.10, "sadness": 0.05}),
                       "venting",
                       r("venting", INTENT_LABELS, {"seeking_empathy": 0.10}),
                       inten))

ANX_SE = [
    ("我最近一直处于一种说不清楚的紧张状态，没有具体原因，就是不安，有人懂吗？", 0.63),
    ("我很担心一件事但又没办法控制，这种无力感很烦，想找人说说。", 0.65),
    ("我心里一直有个事情压着，白天还好，到了晚上就特别明显，需要说出来。", 0.67),
    ("我是那种遇到不确定的事就很难放松的人，最近这种感觉很强，想聊一聊。", 0.62),
    ("这段时间整个人都是绷着的，说不上为什么，就是有人听一听就好。", 0.64),
    ("我有个担心一直放不下，理智上知道可能没事，但就是控制不住，想说说。", 0.66),
    ("我家里最近有个情况让我很不安，不方便细说，就是需要有人陪我说说话。", 0.63),
    ("我一直有一种觉得自己快要撑不住的预感，不知道是不是焦虑，想聊聊。", 0.68),
    ("最近老是有心慌的感觉，没有具体触发点，就是找个地方说出来会好一点。", 0.67),
    ("我在等一个消息，等得心里很不踏实，需要有人陪我等一等。", 0.60),
    ("我最近总是想太多，停不下来，有人理解这种感觉吗？", 0.64),
    ("我有一种说不清的焦虑，可能就是想找人说说，让它不那么堵在心里。", 0.65),
    ("我在担心一件对我来说很重要的事，身边没有合适的人说，想找个地方聊聊。", 0.63),
    ("最近总是容易莫名紧张，有时候心跳会加快，不知道是怎么了，想说说。", 0.66),
    ("我一直有种事情快控制不住的感觉，说不清楚，就是需要说出来。", 0.67),
]
for text, inten in ANX_SE:
    RECORDS.append(rec(text, "anxiety",
                       r("anxiety", EMOTION_LABELS, {"fear": 0.08, "sadness": 0.05}),
                       "seeking_empathy",
                       r("seeking_empathy", INTENT_LABELS, {"venting": 0.14}),
                       inten))


def main() -> None:
    train_path = Path("data/processed/mental_dialogue_train.jsonl")
    with open(train_path, "a", encoding="utf-8") as f:
        for record in RECORDS:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = sum(1 for _ in open(train_path, encoding="utf-8"))
    print(f"Appended {len(RECORDS)} targeted records.  New train size: {total}")

    import collections
    emotion_c: dict = collections.Counter()
    intent_c: dict  = collections.Counter()
    all_r = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            all_r.append(json.loads(line))
    for r2 in all_r:
        emotion_c[r2["emotion_label"]] += 1
        intent_c[r2["intent_label"]]  += 1
    n = len(all_r)
    print("\nEmotion distribution:")
    for lbl, cnt in sorted(emotion_c.items(), key=lambda x: -x[1]):
        print("  %-20s %5d  (%.1f%%)" % (lbl, cnt, cnt/n*100))
    print("\nIntent distribution:")
    for lbl, cnt in sorted(intent_c.items(), key=lambda x: -x[1]):
        print("  %-25s %5d  (%.1f%%)" % (lbl, cnt, cnt/n*100))


if __name__ == "__main__":
    main()
