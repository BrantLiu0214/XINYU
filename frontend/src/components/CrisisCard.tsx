import type { AlertPayload } from '../types';
import styles from './CrisisCard.module.css';

interface Props {
  alert: AlertPayload;
}

export function CrisisCard({ alert }: Props) {
  return (
    <div className={styles.card}>
      <div className={styles.header}>
        ⚠️ 检测到危机信号（{alert.risk_level}）— 以下资源可以帮助您
      </div>
      {alert.resources.length === 0 ? (
        <p className={styles.noResource}>请立即拨打当地心理援助热线或前往最近的医院。</p>
      ) : (
        <ul className={styles.list}>
          {alert.resources.map((r, i) => (
            <li key={i} className={styles.item}>
              <strong>{r.title}</strong>
              {r.phone && <span className={styles.phone}>📞 {r.phone}</span>}
              {r.url && (
                <a className={styles.link} href={r.url} target="_blank" rel="noreferrer">
                  🔗 {r.url}
                </a>
              )}
              {r.description && <span className={styles.desc}>{r.description}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
