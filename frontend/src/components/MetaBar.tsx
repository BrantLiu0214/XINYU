import type { MetaPayload, RiskLevel } from '../types';
import { RiskBadge } from './RiskBadge';
import styles from './MetaBar.module.css';

interface Props {
  meta: MetaPayload;
}

const EMOTION_LABELS: Record<string, string> = {
  anxiety: '焦虑',
  sadness: '悲伤',
  anger: '愤怒',
  fear: '恐惧',
  shame: '羞耻',
  hopelessness: '绝望',
  neutral: '平静',
};

const INTENT_LABELS: Record<string, string> = {
  venting: '情绪宣泄',
  seeking_advice: '寻求建议',
  seeking_empathy: '寻求共情',
  crisis: '危机求助',
  self_disclosure: '自我披露',
  information_seeking: '信息咨询',
};

export function MetaBar({ meta }: Props) {
  return (
    <div className={styles.bar}>
      <span className={styles.tag}>
        {EMOTION_LABELS[meta.emotion] ?? meta.emotion}
      </span>
      <span className={styles.tag}>
        {INTENT_LABELS[meta.intent] ?? meta.intent}
      </span>
      <span className={styles.tag}>
        强度 {(meta.intensity * 100).toFixed(0)}%
      </span>
      <RiskBadge level={meta.risk_level as RiskLevel} small />
    </div>
  );
}
