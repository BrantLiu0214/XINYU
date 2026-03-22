import type { RiskLevel } from '../types';
import styles from './RiskBadge.module.css';

const LABELS: Record<RiskLevel, string> = {
  L0: 'L0 正常',
  L1: 'L1 关注',
  L2: 'L2 警告',
  L3: 'L3 危机',
};

interface Props {
  level: RiskLevel;
  small?: boolean;
}

export function RiskBadge({ level, small }: Props) {
  return (
    <span className={`${styles.badge} ${styles[level]} ${small ? styles.small : ''}`}>
      {LABELS[level]}
    </span>
  );
}
