import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import type { AlertStatus, AlertSummary, DashboardStats, SessionSummary } from '../types';
import { RiskBadge } from '../components/RiskBadge';
import { DashboardCharts } from '../components/DashboardCharts';
import type { EmotionCount, RiskLevelCount } from '../components/DashboardCharts';
import styles from './DashboardPage.module.css';

function formatDate(iso: string) {
  return new Date(iso).toLocaleString('zh-CN', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit',
  });
}

const EMOTION_ZH: Record<string, string> = {
  neutral: '平静', anxiety: '焦虑', sadness: '悲伤',
  anger: '愤怒', fear: '恐惧', shame: '羞耻', hopelessness: '绝望',
};

export function DashboardPage() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [alerts, setAlerts] = useState<AlertSummary[]>([]);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [emotionDist, setEmotionDist] = useState<EmotionCount[]>([]);
  const [riskDist, setRiskDist] = useState<RiskLevelCount[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch('/api/v1/dashboard/sessions').then(r => r.json()),
      fetch('/api/v1/dashboard/alerts').then(r => r.json()),
      fetch('/api/v1/dashboard/stats').then(r => r.json()),
      fetch('/api/v1/dashboard/charts').then(r => r.json()),
    ]).then(([sData, aData, stData, cData]) => {
      setSessions(sData.sessions ?? []);
      setAlerts(aData.alerts ?? []);
      setStats(stData);
      setEmotionDist(cData.emotion_distribution ?? []);
      setRiskDist(cData.risk_distribution ?? []);
      setLoading(false);
    }).catch(err => {
      console.error(err);
      setLoading(false);
    });
  }, []);

  async function updateAlertStatus(alertId: string, newStatus: AlertStatus) {
    const res = await fetch(`/api/v1/dashboard/alerts/${alertId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: newStatus }),
    });
    if (!res.ok) return;
    const updated: AlertSummary = await res.json();
    setAlerts(prev => prev.map(a => a.alert_id === alertId ? updated : a));
  }

  if (loading) return <div className={styles.loading}>加载中…</div>;

  return (
    <div className={styles.page}>
      {/* Stats bar */}
      {stats && (
        <div className={styles.statsBar}>
          <div className={styles.statCard}>
            <div className={styles.statNumber}>{stats.total_sessions}</div>
            <div className={styles.statLabel}>总会话数</div>
          </div>
          <div className={styles.statCard}>
            <div className={styles.statNumber}>{stats.total_messages}</div>
            <div className={styles.statLabel}>总消息数</div>
          </div>
          <div className={styles.statCard}>
            <div className={stats.open_alerts > 0 ? styles.statNumberAlert : styles.statNumber}>
              {stats.open_alerts}
            </div>
            <div className={styles.statLabel}>待处理预警</div>
          </div>
          <div className={styles.statCard}>
            <div className={stats.l3_alerts > 0 ? styles.statNumberAlert : styles.statNumber}>
              {stats.l3_alerts}
            </div>
            <div className={styles.statLabel}>L3 危机预警</div>
          </div>
        </div>
      )}

      {/* Visualisation charts */}
      <section className={styles.section}>
        <h2 className={styles.heading}>数据可视化</h2>
        <DashboardCharts emotionData={emotionDist} riskData={riskDist} />
      </section>

      <section className={styles.section}>
        <h2 className={styles.heading}>会话列表 ({sessions.length})</h2>
        {sessions.length === 0 ? (
          <p className={styles.empty}>暂无会话记录</p>
        ) : (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>会话 ID</th>
                <th>开始时间</th>
                <th>消息数</th>
                <th>主要情绪</th>
                <th>最高风险</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map(s => (
                <tr key={s.session_id}>
                  <td>
                    <Link to={`/session/${s.session_id}`} className={styles.link}>
                      {s.session_id.slice(0, 8)}…
                    </Link>
                  </td>
                  <td>{formatDate(s.started_at)}</td>
                  <td>{s.message_count}</td>
                  <td className={styles.emotionCell}>
                    {s.dominant_emotion ? (EMOTION_ZH[s.dominant_emotion] ?? s.dominant_emotion) : '—'}
                  </td>
                  <td><RiskBadge level={s.latest_risk_level} small /></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className={styles.section}>
        <h2 className={styles.heading}>预警记录 ({alerts.length})</h2>
        {alerts.length === 0 ? (
          <p className={styles.empty}>暂无预警记录</p>
        ) : (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>时间</th>
                <th>会话</th>
                <th>风险等级</th>
                <th>原因</th>
                <th>状态 / 操作</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map(a => (
                <tr key={a.alert_id}>
                  <td>{formatDate(a.created_at)}</td>
                  <td>
                    <Link to={`/session/${a.session_id}`} className={styles.link}>
                      {a.session_id.slice(0, 8)}…
                    </Link>
                  </td>
                  <td><RiskBadge level={a.risk_level} small /></td>
                  <td className={styles.reasons}>{a.reasons.join('；')}</td>
                  <td className={styles.actionCell}>
                    {a.status === 'resolved' ? (
                      <span className={`${styles.status} ${styles.resolved}`}>已解决</span>
                    ) : (
                      <div className={styles.actionBtns}>
                        {a.status === 'open' && (
                          <button
                            className={`${styles.actionBtn} ${styles.actionBtnAck}`}
                            onClick={() => updateAlertStatus(a.alert_id, 'acknowledged')}
                          >
                            确认
                          </button>
                        )}
                        <button
                          className={`${styles.actionBtn} ${styles.actionBtnResolve}`}
                          onClick={() => updateAlertStatus(a.alert_id, 'resolved')}
                        >
                          解决
                        </button>
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
